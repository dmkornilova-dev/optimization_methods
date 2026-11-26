from math import sqrt
from helping.log import make_logger
from helping.types import MultivarResult
from helping.line_search import line_search


def dfp_method(f, grad, x0, eps1, eps2, M, step_method, svenn_t, verbose=False):
    if eps1 <= 0 or eps2 <= 0:
        raise ValueError("eps1 и eps2 должны быть положительными числами")
    if M <= 0:
        raise ValueError("M должно быть положительным числом")
    if svenn_t <= 0:
        raise ValueError("Шаг метода Свенна должен быть положительным числом")

    log = make_logger(verbose)

    # шаг 1
    x = list(x0)
    log("Метод Дэвидона-Флетчера-Пауэлла (ДФП) - подробное решение\n")
    log("Шаг 1. Задаём:")
    log(f"x0 = {x}")
    log(f"eps1 = {eps1}")
    log(f"eps2 = {eps2}")
    log(f"M = {M}")

    # шаг 2: A^0 = E (единичная матрица)
    k = 0
    n = len(x)
    A = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)] # единичная матрица
    log(f"\nШаг 2. Положим k = 0, A0 = E = {A}.")

    fx = f(*x)  # для критерия 2 в шаге 13
    n_evals = 1  # f(x0)
    prev_ok = False  # выполнены ли оба условия шага 13 на предыдущей итерации
    x_prev = None  # x^{k-1} для разности аргументов
    g_prev = None  # grad f(x^{k-1}) для разности градиентов
    history = []

    while True:
        # шаг 3
        g = grad(*x)  # частные производные при данном x=(x1;x2)
        log(f"\nШаг 3 (итерация k = {k}). x{k} = {x}, f(x{k}) = {fx:.6f}")
        log(f"grad f(x{k}) = {g}")

        # шаг 4
        norm_g = sqrt(sum(gi ** 2 for gi in g))
        log(f"\nШаг 4. ||grad f(x{k})|| = {norm_g:.6f}")
        if norm_g <= eps1:
            reason = "step4"
            log(f"{norm_g:.6f} <= eps1 = {eps1} - алгоритм завершён, x* = x{k}.")
            break
        log(f"{norm_g:.6f} > eps1 = {eps1} - переходим к шагу 5.")

        # шаг 5
        if k >= M:
            reason = "step5"
            log(f"\nШаг 5. k = {k} >= M = {M} - алгоритм завершён, x* = x{k}.")
            break
        if k == 0:
            log(f"\nШаг 5. k = {k} < M = {M}, k = 0 - переходим к шагу 10.")
        else:
            log(f"\nШаг 5. k = {k} < M = {M}, k >= 1 - переходим к шагу 6.")

        if k >= 1:
            # шаг 6 - изменение градиента: delta_g^k = grad f(x^k) - grad f(x^{k-1})
            delta_g = [gi - gpi for gi, gpi in zip(g, g_prev)]
            log(f"\nШаг 6. delta_g{k} = grad f(x{k}) - grad f(x{k - 1}) = {delta_g}")

            # шаг 7 - изменение аргумента: delta_x^k = x^k - x^{k-1}
            delta_x = [xi - xpi for xi, xpi in zip(x, x_prev)]
            log(f"\nШаг 7. delta_x{k} = x{k} - x{k - 1} = {delta_x}")

            # шаг 8 - поправка ДФП:
            # A_c = delta_x delta_x^T / (delta_x^T delta_g)
            #       - A delta_g delta_g^T A / (delta_g^T A delta_g)
            Ag = [sum(A[i][j] * delta_g[j] for j in range(n)) for i in range(n)]
            denom1 = sum(delta_x[i] * delta_g[i] for i in range(n))
            denom2 = sum(delta_g[i] * Ag[i] for i in range(n))

            A_c = [[delta_x[i] * delta_x[j] / denom1 - Ag[i] * Ag[j] / denom2
                    for j in range(n)] for i in range(n)]
            log(f"\nШаг 8. (delta_x)^T delta_g = {denom1:.6f}, "
                f"(delta_g)^T A delta_g = {denom2:.6f}")
            log(f"A_c{k} = {A_c}")

            # шаг 9 - обновление матрицы
            A = [[A[i][j] + A_c[i][j] for j in range(n)] for i in range(n)]
            log(f"\nШаг 9. A{k} = A{k - 1} + A_c{k} = {A}")

        # шаг 10 - направление спуска
        d = [-sum(A[i][j] * g[j] for j in range(n)) for i in range(n)]
        log(f"\nШаг 10. d{k} = -A{k} * grad f(x{k}) = {d}")

        # шаг 11: phi(tk) = f(x^k + t_k * d^k) - ищем точку минимума phi
        def phi(t, x=x, d=d):
            return f(*[xi + t * di for xi, di in zip(x, d)])
        t, n_line = line_search(phi, step_method, svenn_t, verbose=verbose)
        log(f"\nШаг 11. t{k}* = {t:.6f} (вычислений функции при поиске шага: {n_line})")

        # шаг 12 - новая точка
        x_new = [xi + t * di for xi, di in zip(x, d)]
        f_new = f(*x_new)
        n_evals += n_line + 1  # ОБЩЕЕ количество вычислений функции
        log(f"\nШаг 12. x{k + 1} = x{k} + t{k}* * d{k} = {x_new}")
        log(f"f(x{k + 1}) = {f_new:.6f}")

        # шаг 13 - проверка двух критериев
        norm_x = sqrt(sum((a - b) ** 2 for a, b in zip(x_new, x)))
        mod_fx = abs(f_new - fx)
        ok = norm_x < eps2 and mod_fx < eps2
        log(f"\nШаг 13. ||x{k + 1} - x{k}|| = {norm_x:.6f}, |f(x{k + 1}) - f(x{k})| = {mod_fx:.6f}")

        history.append({
            "k": k,
            "x_k": x,
            "f_xk": fx,
            "grad_norm": norm_g,
            "direction": "-grad" if k == 0 else "DFP",
            "A_k": [row[:] for row in A],
            "d": d,
            "t_k": t,
            "x_k1": x_new,
            "f_xk1": f_new,
            "norm_x": norm_x,
            "mod_fx": mod_fx,
            "ok": ok,
        })

        x_prev, g_prev = x, g
        x, fx = x_new, f_new

        if ok and prev_ok:
            reason = "step13"
            log(f"Оба условия выполнены при k = {k} и k = {k - 1} - алгоритм завершён, "
                f"x* = x{k + 1}.")
            break
        if ok:
            log(f"Оба условия выполнены при k = {k}, но не при k = {k - 1} - "
                f"полагаем k = {k + 1} и переходим к шагу 3.")
        else:
            log(f"Хотя бы одно условие не выполнено - полагаем k = {k + 1} "
                f"и переходим к шагу 3.")
        prev_ok = ok
        k += 1

    log(f"\nx* = {x}, f(x*) = {fx:.6f}")

    return MultivarResult(
        x=x, f_x=fx,
        n_iter=len(history),
        n_evals=n_evals,
        history=history,
        extra={"reason": reason, "A_final": A},
    )
