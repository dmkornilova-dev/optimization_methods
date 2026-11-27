from math import sqrt
from helping.var9 import (f2 as f, grad_f2 as grad, hess_f2 as hess, # целевая функция, ее градиент и гессиан
                          g1, grad_g1, hess_g1, # ограничение-равенство, его градиент и гессиан
                          X0, # начальная точка
                          A, B, C, D, E, F, # коэффициенты целевой функции
                          U, V, W, # коэффициенты ограничения
                          X_MIN_CONSTRAINED) # аналитический условный минимум
from helping.print_tables import print_table_constrained

# методы одномерной минимизации
from methods.lab3.dichotomy import dichotomy_method
from methods.lab3.golden_section import golden_section_method
from methods.lab3.fibonacci import fibonacci_method

# методы многомерной минимизации
from methods.lab4.steepest_descent import steepest_descent_method
from methods.lab4.fletcher_reeves import fletcher_reeves_method
from methods.lab4.dfp import dfp_method
from methods.lab4.newton import newton_method
from methods.lab4.newton_raphson import newton_raphson_method
from methods.lab4.marquardt import marquardt_method

# основной метод (задание лабы 5)
from methods.lab5.penalty import penalty_method


# параметры метода множителей штрафов
R0, C_PENALTY, EPS = 1, 10, 0.0001
EQUALITIES = [(g1, grad_g1, hess_g1)] # ограничения-равенства
INEQUALITIES = [] # ограничения-неравенства
M_OUTER = 30

# параметры внутренней минимизации
EPS1, EPS2, M = 0.00001, 0.00000001, 10000
MARQUARDT_MU_0 = 140  # при необходимости Марквардт сам увеличивает mu
SVENN_T = 0.1
VERBOSE = False

STEP_METHODS = {
    "Дихотомия": dict(func=dichotomy_method, l=0.0000001, eps=0.00000001),
    "Золотое сечение": dict(func=golden_section_method, l=0.0000001),
    "Фибоначчи": dict(func=fibonacci_method, l=0.0000001, eps=0.00000001),
}
MIN_METHODS = {
    "Наискорейший спуск": dict(func=steepest_descent_method),
    "Ньютон": dict(func=newton_method, use_hess=True),
    "Ньютон-Рафсон": dict(func=newton_raphson_method, use_hess=True),
    "Флетчер-Ривс": dict(func=fletcher_reeves_method),
    "Марквардт": dict(func=marquardt_method, use_hess=True),
    "ДФП": dict(func=dfp_method),
}


def run_method(name, step_name=None):
    params = dict(MIN_METHODS[name])
    params["eps1"] = EPS1
    params["M"] = M

    if name == "Марквардт":
        params["mu0"] = MARQUARDT_MU_0
    else:
        params["eps2"] = EPS2
        params["step_method"] = STEP_METHODS[step_name]
        params["svenn_t"] = SVENN_T

    print(f"Метод штрафов: f(x) = {A}*x1^2 + {B}*x2^2 + {C}*x1*x2 + {D}*x1 + {E}*x2 + {F}")
    print(f"g1(x) = {U}*x1 + {V}*x2 - ({W}) = 0; x0 = {X0}; r0 = {R0}; C = {C_PENALTY}; eps = {EPS}")
    print(f"Метод многомерной минимизации: {name}; eps1 = {EPS1}, M = {M}")
    if step_name is not None:
        step_params = {k: v for k, v in STEP_METHODS[step_name].items() if k != "func"}
        print(f"eps2 = {EPS2}; поиск шага: Свенн (t = {SVENN_T}) + {step_name}, параметры: {step_params}")
        if name == "Ньютон":
            print("Поиск шага только при переходе к антиградиенту; иначе t_k = 1.")
    else:
        print(f"Линейный поиск не используется; mu0 = {MARQUARDT_MU_0}")

    res = penalty_method(f, grad, hess, X0, EQUALITIES, INEQUALITIES,
                          R0, C_PENALTY, EPS, min_method=params,
                          M=M_OUTER, verbose=VERBOSE)
    print_table_constrained(res.history)
    print(f"Найденная точка: x* = ({res.x[0]:.6f}; {res.x[1]:.6f}), f(x*) = {res.f_x:.6f}")
    print(f"g1(x*) = {res.history[-1]['g_eq'][0]:.6e}, P(x*) = {res.extra['P']:.6e}")
    if res.extra["reason"] == "step4":
        print("Причина остановки: выполнен критерий P(x*) <= eps (шаг 4)")
    else:
        print("Причина остановки: достигнуто предельное число внешних итераций. ")
    if any(row["inner_reason"] == "step5" for row in res.history):
        print("Внимание: внутренняя минимизация достигла предельного числа итераций: ")
        print("внутренние вычисления метода многомерной оптимизации могут быть неточны. ")
    print(f"Внешних итераций: {res.n_iter}, внутренних итераций: {res.extra['inner_iter']}")
    print(f"Вычислений F: {res.extra['F_evals']}, вычислений f (включая F): {res.n_evals}")
    return res


def print_comparison(results):
    f_min = f(*X_MIN_CONSTRAINED) # значение функции в аналитическом условном минимуме
    print("Сравнение сочетаний методов")
    print(f"Аналитический условный минимум: x_min = ({X_MIN_CONSTRAINED[0]:.6f}; {X_MIN_CONSTRAINED[1]:.6f}), f(x_min) = {f_min:.6f}")
    print(f"{'Метод лабы 4':<20} | {'Метод лабы 3':<16} | {'x*':^24} | {'f(x*)':>10} | "
          f"{'||x*-x_min||':>13} | {'|f(x*)-f_min|':>14} | {'|g(x*)|':>10} | {'P':>10} | "
          f"{'Внеш. ит.':>9} | {'Внутр. ит.':>10} | {'Выч. F':>8} | {'Выч. f':>8} | Статус")
    for (name, step_name), res in results.items():
        if res is None:
            print(f"{name:<20} | {step_name or '—':<16} | ошибка")
            continue

        # форматирование найденной точки x*
        x_str = f"({res.x[0]:.6f}; {res.x[1]:.6f})"

        # евклидово расстояние между аналитическим минимумом и найденным x*
        dist = sqrt(sum((a-b)**2 for a, b in zip(res.x, X_MIN_CONSTRAINED)))

        status = "критерий P" if res.extra["reason"] == "step4" else "лимит внеш."
        if any(row["inner_reason"] == "step5" for row in res.history):
            status += "; лимит внутр."

        print(f"{name:<20} | {step_name or '—':<16} | {x_str:^24} | {res.f_x:>10.6f} | "
              f"{dist:>13.3e} | {abs(res.f_x-f_min):>14.3e} | {res.extra['max_err']:>10.3e} | "
              f"{res.extra['P']:>10.3e} | {res.n_iter:>9} | {res.extra['inner_iter']:>10} | "
              f"{res.extra['F_evals']:>8} | {res.n_evals:>8} | {status}")


if __name__ == "__main__":
    results = {} # словарь: ключ - пара (метод лабы 4; метод лабы 3); значение - результат penalty_method
    for name in MIN_METHODS:
        step_names = [None] if name == "Марквардт" else STEP_METHODS
        for step_name in step_names:
            try:
                results[name, step_name] = run_method(name, step_name)
            except (ValueError, ArithmeticError, RuntimeError) as error:
                print(f"Ошибка: {error}")
                results[name, step_name] = None
            print("\n")
    print_comparison(results)
