"""
Лабораторная работа: «Изучение и применение алгоритмов обучения с подкреплением
для решения прикладной задачи».

Тема: управление светофором на перекрёстке с помощью Q-обучения (Q-learning).

Запуск:
    python3 traffic_light_qlearning.py

Результаты (графики, анимация, числовые итоги) сохраняются в папку results/.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


# =============================================================================
# 1. СРЕДА: перекрёсток двух дорог (Север–Юг и Запад–Восток)
# =============================================================================
class TrafficIntersectionEnv:
    """
    Упрощённая модель перекрёстка.

    * Есть два направления движения: NS (Север–Юг) и EW (Запад–Восток).
    * Один шаг моделирования = 5 секунд реального времени.
    * На каждом шаге к перекрёстку подъезжают машины (поток Пуассона).
      Интенсивность меняется в течение эпизода: первую половину загружено
      направление NS («утренний час пик»), вторую — направление EW.
    * Светофор имеет две фазы: 0 — зелёный для NS, 1 — зелёный для EW.
    * За шаг на зелёный проезжает не более SATURATION_FLOW машин.
    * При переключении фазы включается жёлтый сигнал: в течение этого шага
      не проезжает никто (потерянное время), это «цена» переключения.
    """

    STEP_SECONDS = 5
    EPISODE_STEPS = 200          # 200 шагов * 5 с ≈ 16,7 минуты
    SATURATION_FLOW = 2          # машин за шаг на зелёный сигнал
    MAX_QUEUE = 50               # вместимость подъезда (длина дороги)

    # Интенсивности прибытия машин (среднее число машин за шаг)
    RATES_FIRST_HALF = (0.9, 0.3)    # (NS, EW) — час пик по NS
    RATES_SECOND_HALF = (0.3, 0.9)   # (NS, EW) — час пик по EW

    # Границы дискретизации длины очереди: 0 | 1-2 | 3-5 | 6-9 | 10-14 | 15-20 | 21+
    QUEUE_BINS = np.array([1, 3, 6, 10, 15, 21])
    N_QUEUE_LEVELS = len(QUEUE_BINS) + 1     # 7 уровней
    N_TIME_LEVELS = 4                        # время в фазе: 0, 1, 2, 3+ шагов
    N_PHASES = 2

    N_STATES = N_QUEUE_LEVELS * N_QUEUE_LEVELS * N_PHASES * N_TIME_LEVELS  # 392
    N_ACTIONS = 2   # 0 — оставить текущую фазу, 1 — переключить фазу

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Начало нового эпизода: пустой перекрёсток, зелёный для NS."""
        self.queues = np.zeros(2, dtype=int)   # [очередь NS, очередь EW]
        self.phase = 0
        self.time_in_phase = 0
        self.t = 0
        self.total_arrived = 0
        self.total_passed = 0
        self.yellow = False
        return self.get_state()

    def current_rates(self):
        return self.RATES_FIRST_HALF if self.t < self.EPISODE_STEPS // 2 else self.RATES_SECOND_HALF

    def get_state(self):
        """
        Наблюдаемое агентом состояние s = (уровень очереди NS, уровень очереди EW,
        текущая фаза, время в фазе), закодированное одним целым числом —
        номером строки в Q-таблице.
        """
        q_ns = int(np.digitize(self.queues[0], self.QUEUE_BINS))
        q_ew = int(np.digitize(self.queues[1], self.QUEUE_BINS))
        tp = min(self.time_in_phase, self.N_TIME_LEVELS - 1)
        return ((q_ns * self.N_QUEUE_LEVELS + q_ew) * self.N_PHASES + self.phase) * self.N_TIME_LEVELS + tp

    def step(self, action):
        """
        Выполняет действие агента и моделирует 5 секунд движения.
        Возвращает (новое состояние, награда, флаг окончания эпизода, доп. информация).
        """
        # --- 1) Применяем действие светофора ---
        if action == 1:
            self.phase = 1 - self.phase
            self.time_in_phase = 0
            self.yellow = True       # шаг жёлтого сигнала: никто не проезжает
        else:
            self.time_in_phase += 1
            self.yellow = False

        # --- 2) Машины на зелёный сигнал проезжают перекрёсток ---
        passed = 0
        if not self.yellow:
            passed = min(self.queues[self.phase], self.SATURATION_FLOW)
            self.queues[self.phase] -= passed
        self.total_passed += passed

        # --- 3) Прибывают новые машины ---
        arrivals = self.rng.poisson(self.current_rates())
        self.queues = np.minimum(self.queues + arrivals, self.MAX_QUEUE)
        self.total_arrived += int(arrivals.sum())

        # --- 4) Награда: минус число машин, стоящих в очередях ---
        # Сумма очередей за эпизод пропорциональна суммарному времени ожидания,
        # поэтому максимизация награды = минимизация задержек водителей.
        reward = -float(self.queues.sum())

        self.t += 1
        done = self.t >= self.EPISODE_STEPS   # условие завершения эпизода
        info = {"passed": passed, "arrivals": arrivals}
        return self.get_state(), reward, done, info


# =============================================================================
# 2. АГЕНТ: табличное Q-обучение
# =============================================================================
class QLearningAgent:
    """
    Q-обучение: агент хранит таблицу Q(s, a) — оценку ожидаемой суммарной
    (дисконтированной) награды при выполнении действия a в состоянии s
    и дальнейшем оптимальном поведении.

    Правило обновления (уравнение Беллмана):
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay_episodes=3000, seed=None):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha                  # скорость обучения
        self.gamma = gamma                  # коэффициент дисконтирования
        self.epsilon = epsilon_start        # вероятность случайного действия (исследование)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, greedy=False):
        """ε-жадная стратегия: с вероятностью ε — случайное действие, иначе лучшее по Q."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q_table[state]
        best = np.flatnonzero(q == q.max())        # при равенстве — случайный из лучших
        return int(self.rng.choice(best))

    def update(self, s, a, r, s_next, done):
        """Одно обновление Q-таблицы по наблюдённому переходу (s, a, r, s')."""
        target = r if done else r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.alpha * (target - self.q_table[s, a])

    def decay_epsilon(self, episode):
        """Линейное уменьшение ε: сначала агент много исследует, потом использует опыт."""
        frac = min(1.0, episode / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


# =============================================================================
# 3. ОБУЧЕНИЕ
# =============================================================================
def train(n_episodes=5000):
    env = TrafficIntersectionEnv(seed=SEED)
    agent = QLearningAgent(env.N_STATES, env.N_ACTIONS, seed=SEED)
    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon(ep + 1)
        episode_rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            last = np.mean(episode_rewards[-100:])
            print(f"Эпизод {ep + 1:5d} | средняя награда за 100 эпизодов: {last:9.1f} | ε = {agent.epsilon:.3f}")

    return agent, np.array(episode_rewards)


# =============================================================================
# 4. ОЦЕНКА И СРАВНЕНИЕ С БАЗОВЫМИ СТРАТЕГИЯМИ
# =============================================================================
def run_episode(env, policy, record=False):
    """
    Прогоняет один эпизод с заданной стратегией policy(env, state) -> action.
    Возвращает награду, среднюю задержку на машину (с) и, при record=True, историю.
    """
    state = env.reset()
    total_reward, done = 0.0, False
    history = {"q_ns": [], "q_ew": [], "phase": [], "yellow": [], "action": []}
    while not done:
        action = policy(env, state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if record:
            history["q_ns"].append(int(env.queues[0]))
            history["q_ew"].append(int(env.queues[1]))
            history["phase"].append(env.phase)
            history["yellow"].append(env.yellow)
            history["action"].append(action)
    # Суммарное время ожидания = сумма очередей по шагам * длительность шага
    avg_delay = -total_reward * env.STEP_SECONDS / max(env.total_arrived, 1)
    return total_reward, avg_delay, history


def fixed_time_policy(green_steps):
    """Классический светофор с фиксированным циклом: переключение каждые green_steps шагов."""
    def policy(env, state):
        return 1 if env.time_in_phase >= green_steps - 1 else 0
    return policy


def random_policy(rng):
    def policy(env, state):
        return int(rng.integers(2))
    return policy


def evaluate(policy, n_episodes=100, seed=1000):
    env = TrafficIntersectionEnv(seed=seed)   # одинаковый поток машин для всех стратегий
    rewards, delays = [], []
    for _ in range(n_episodes):
        r, d, _ = run_episode(env, policy)
        rewards.append(r)
        delays.append(d)
    return float(np.mean(rewards)), float(np.mean(delays))


# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# =============================================================================
def plot_learning_curve(rewards, baseline_reward, path):
    window = 100
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    block_avg = rewards[: len(rewards) // window * window].reshape(-1, window).mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    ax = axes[0]
    ax.plot(rewards, color="lightsteelblue", linewidth=0.6, label="Награда за эпизод")
    ax.plot(np.arange(window - 1, len(rewards)), moving_avg, color="navy", linewidth=2,
            label=f"Скользящее среднее ({window} эпизодов)")
    ax.axhline(baseline_reward, color="crimson", linestyle="--",
               label="Лучший светофор с фиксированным циклом")
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Суммарная награда")
    ax.set_title("Обучение агента: награда за эпизод")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(np.arange(len(block_avg)) * window + window / 2, block_avg, width=window * 0.85, color="teal")
    ax.axhline(baseline_reward, color="crimson", linestyle="--", label="Фиксированный цикл")
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Средняя награда")
    ax.set_title("Средняя награда за каждые 100 эпизодов")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_episode_comparison(hist_agent, hist_fixed, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, hist, title in [(axes[0], hist_agent, "Обученный агент (Q-learning)"),
                            (axes[1], hist_fixed, "Светофор с фиксированным циклом")]:
        t = np.arange(len(hist["q_ns"]))
        # Цветом фона показываем, какому направлению горит зелёный
        for i, (ph, yl) in enumerate(zip(hist["phase"], hist["yellow"])):
            color = "gold" if yl else ("#d6eaf8" if ph == 0 else "#fadbd8")
            ax.axvspan(i - 0.5, i + 0.5, color=color, linewidth=0)
        ax.plot(t, hist["q_ns"], color="navy", label="Очередь Север–Юг")
        ax.plot(t, hist["q_ew"], color="darkred", label="Очередь Запад–Восток")
        ax.axvline(len(t) / 2, color="black", linestyle=":", linewidth=1)
        total = sum(hist["q_ns"]) + sum(hist["q_ew"])
        ax.set_title(f"{title}: сумма очередей за эпизод = {total}")
        ax.set_ylabel("Машин в очереди")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Шаг (5 с). Фон: голубой — зелёный NS, розовый — зелёный EW, жёлтый — переключение")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy(agent, path):
    """Тепловая карта выученной стратегии: переключать ли фазу при данных очередях."""
    env = TrafficIntersectionEnv
    labels = ["0", "1-2", "3-5", "6-9", "10-14", "15-20", "21+"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tp = env.N_TIME_LEVELS - 1   # фаза горит уже 3+ шага
    for phase, ax in enumerate(axes):
        grid = np.zeros((env.N_QUEUE_LEVELS, env.N_QUEUE_LEVELS))
        for q_ns in range(env.N_QUEUE_LEVELS):
            for q_ew in range(env.N_QUEUE_LEVELS):
                s = ((q_ns * env.N_QUEUE_LEVELS + q_ew) * env.N_PHASES + phase) * env.N_TIME_LEVELS + tp
                grid[q_ns, q_ew] = np.argmax(agent.q_table[s]) if agent.q_table[s].any() else np.nan
        ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower")
        for i in range(env.N_QUEUE_LEVELS):
            for j in range(env.N_QUEUE_LEVELS):
                txt = "—" if np.isnan(grid[i, j]) else ("ПЕРЕКЛ." if grid[i, j] == 1 else "держать")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        ax.set_xticks(range(env.N_QUEUE_LEVELS), labels)
        ax.set_yticks(range(env.N_QUEUE_LEVELS), labels)
        ax.set_xlabel("Очередь Запад–Восток")
        ax.set_ylabel("Очередь Север–Юг")
        ax.set_title(f"Сейчас зелёный для {'Север–Юг' if phase == 0 else 'Запад–Восток'}")
    fig.suptitle("Выученная стратегия агента (фаза длится 3+ шага; «—» — состояние не встречалось)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_animation(hist, path, n_frames=80):
    """GIF-анимация работы обученного агента: перекрёсток, очереди и сигналы светофора."""
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(i):
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((-10, -10), 20, 20, color="#7dcea0"))
        ax.add_patch(Rectangle((-1.5, -10), 3, 20, color="#555"))   # дорога NS
        ax.add_patch(Rectangle((-10, -1.5), 20, 3, color="#555"))   # дорога EW

        q_ns, q_ew = hist["q_ns"][i], hist["q_ew"][i]
        phase, yellow = hist["phase"][i], hist["yellow"][i]

        # Машины в очереди NS (едут с юга на север, стоят ниже перекрёстка)
        for k in range(min(q_ns, 16)):
            ax.add_patch(Rectangle((-0.6, -2.3 - k * 0.5), 1.2, 0.4, color="#2e86c1"))
        # Машины в очереди EW (едут с запада на восток, стоят левее перекрёстка)
        for k in range(min(q_ew, 16)):
            ax.add_patch(Rectangle((-2.3 - k * 0.5, -0.6), 0.4, 1.2, color="#cb4335"))

        def light(on):
            return "gold" if yellow else ("lime" if on else "red")
        ax.add_patch(Circle((2.3, -2.3), 0.5, color=light(phase == 0), ec="black"))
        ax.add_patch(Circle((-2.3, 2.3), 0.5, color=light(phase == 1), ec="black"))
        ax.text(2.9, -2.4, "NS", fontsize=9)
        ax.text(-3.9, 2.9, "EW", fontsize=9)
        ax.text(-9.5, 9, f"Шаг {i}  (t = {i * TrafficIntersectionEnv.STEP_SECONDS} с)", fontsize=11, weight="bold")
        ax.text(-9.5, 8, f"Очередь NS: {q_ns}", color="#1b4f72", fontsize=10)
        ax.text(-9.5, 7.2, f"Очередь EW: {q_ew}", color="#78281f", fontsize=10)
        ax.text(-9.5, 6.4, "Действие: " + ("ПЕРЕКЛЮЧИТЬ" if hist["action"][i] else "держать"), fontsize=10)

    start = TrafficIntersectionEnv.EPISODE_STEPS // 2 - n_frames // 2   # момент смены часа пик
    frames = range(start, start + n_frames)
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=300)
    anim.save(path, writer=animation.PillowWriter(fps=3))
    plt.close(fig)


# =============================================================================
# 6. ГЛАВНАЯ ПРОГРАММА
# =============================================================================
def main():
    print("=== Обучение агента Q-learning ===")
    agent, rewards = train(n_episodes=5000)
    np.save(os.path.join(RESULTS_DIR, "q_table.npy"), agent.q_table)

    print("\n=== Оценка стратегий на 100 тестовых эпизодах ===")
    results = {}
    agent_policy = lambda env, s: agent.choose_action(s, greedy=True)
    results["Q-learning агент"] = evaluate(agent_policy)

    best_fixed_name, best_fixed = None, None
    for g in range(2, 13):
        res = evaluate(fixed_time_policy(g))
        results[f"Фиксированный цикл, зелёный {g * 5} с"] = res
        if best_fixed is None or res[0] > best_fixed[0]:
            best_fixed_name, best_fixed, best_g = f"Фиксированный цикл, зелёный {g * 5} с", res, g
    results["Случайное переключение"] = evaluate(random_policy(np.random.default_rng(SEED)))

    for name, (r, d) in results.items():
        print(f"{name:40s} | средняя награда {r:9.1f} | средняя задержка {d:6.1f} с/машину")

    agent_r, agent_d = results["Q-learning агент"]
    print(f"\nЛучший фиксированный цикл: {best_fixed_name}")
    print(f"Сокращение средней задержки агентом: {(1 - agent_d / best_fixed[1]) * 100:.1f} %")

    # Графики и анимация
    plot_learning_curve(rewards, best_fixed[0], os.path.join(RESULTS_DIR, "learning_curve.png"))
    _, _, hist_agent = run_episode(TrafficIntersectionEnv(seed=7), agent_policy, record=True)
    _, _, hist_fixed = run_episode(TrafficIntersectionEnv(seed=7), fixed_time_policy(best_g), record=True)
    plot_episode_comparison(hist_agent, hist_fixed, os.path.join(RESULTS_DIR, "episode_comparison.png"))
    plot_policy(agent, os.path.join(RESULTS_DIR, "policy.png"))
    make_animation(hist_agent, os.path.join(RESULTS_DIR, "agent_animation.gif"))

    summary = {
        "results": {k: {"avg_reward": v[0], "avg_delay_s": v[1]} for k, v in results.items()},
        "best_fixed": best_fixed_name,
        "delay_reduction_percent": (1 - agent_d / best_fixed[1]) * 100,
        "first_100_avg_reward": float(rewards[:100].mean()),
        "last_100_avg_reward": float(rewards[-100:].mean()),
        "visited_states": int((agent.q_table != 0).any(axis=1).sum()),
        "total_states": TrafficIntersectionEnv.N_STATES,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nГрафики и анимация сохранены в папку: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
