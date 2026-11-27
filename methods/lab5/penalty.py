from helping.log import make_logger
from helping.types import MultivarResult


def penalty_method(f, grad, hess, x0, equalities, inequalities,
                    r0, C, eps, min_method, M=100, verbose=False):
    # каждое ограничение задаётся тройкой (g, grad_g, hess_g)
    # equalities: g(x) = 0; inequalities: g(x) <= 0
    if r0 <= 0 or C <= 1 or eps <= 0 or M <= 0:
        raise ValueError("Требуется r0 > 0, C_PENALTY > 1, eps > 0, M > 0")
    log = make_logger(verbose)
    params = dict(min_method)
    method = params.pop("func")
    use_hess = params.pop("use_hess", False) #требуется ли выбранному методу матрица Гессе

    # шаг 1
    x = list(x0)
    r = r0 # параметр штрафа
    k = 0
    history = []
    n_evals = 0 # количество вычислений целевой функции
    n_inner = 0 # суммарное количество внутренних итераций
    n_F = 0 # количество вычислений вспомогательной функции
    log("Метод штрафов - подробное решение\n")
    log(f"Шаг 1. x0 = {x}, r0 = {r}, C_PENALTY = {C}, eps = {eps}, k = 0")

    while True:
        # шаг 2 - вспомогательная функция F = f + P
        def P(*point): # функция штрафа
            p_eq = r / 2 * sum(g(*point)**2 for g, _, _ in equalities)
            p_ineq = r / 2 * sum(max(0, g(*point))**2 for g, _, _ in inequalities)
            return p_eq + p_ineq

        def F(*point): # вспомогательная функция
            nonlocal n_evals, n_F
            n_evals += 1
            n_F += 1
            return f(*point) + P(*point)

        def grad_F(*point): # для методов лабы 4: градиент функции, которую минимизируем
            result = list(grad(*point))
            for g, gg, _ in equalities:
                coef = r*g(*point)
                result = [a + coef*b for a, b in zip(result, gg(*point))]
            for g, gg, _ in inequalities:
                coef = max(0, r*g(*point))
                result = [a + coef*b for a, b in zip(result, gg(*point))]
            return result

        def hess_F(*point): # hess_F = hess(f) + sum(r * grad(g) * grad(g)^T + r*g * H(g))
            result = [list(row) for row in hess(*point)] # первое слагаемое - гессиан целевой функции
            # ограничения-равенства
            for g, gg, hg in equalities:
                coef = r * g(*point) # для второй части слагаемого
                dg = gg(*point) # градиент ограничения для первой части слагаемого
                H = hg(*point) # гессиан ограничения для второй части слагаемого
                for i in range(len(point)):
                    for j in range(len(point)):
                        result[i][j] += r * dg[i] * dg[j] # r * grad(g) * grad(g)^T - первая часть слагаемого
                        result[i][j] += coef * H[i][j] # r*g * H(g) - вторая часть слагаемого

            # ограничения-неравенства
            for g, gg, hg in inequalities:
                coef = r * g(*point)
                # обе поправки учитываются только при g > 0
                if coef > 0:
                    dg = gg(*point)
                    H = hg(*point)
                    for i in range(len(point)):
                        for j in range(len(point)):
                            result[i][j] += r * dg[i] * dg[j]
                            result[i][j] += coef * H[i][j]

            return result

        log(f"\nШаг 2 (итерация k = {k}). x{k} = {x}, r{k} = {r}")
        log("F(x) = f(x) + P(x)")
        if verbose:
            terms = ["f(x)"]
            for j in range(1, len(equalities) + 1):
                terms.append(f"{r / 2}*g{j}(x)^2")
            for j in range(len(equalities) + 1, len(equalities) + len(inequalities) + 1):
                terms.append(f"{r / 2}*max(0, g{j}(x))^2")
            log("F(x) = " + " + ".join(terms))

        # шаг 3 - безусловная минимизация F методом из лабы 4
        log(f"\nШаг 3. Минимизируем F методом {method.__name__}, начальная точка: {x}")
        if use_hess:
            res = method(F, grad_F, hess_F, x, verbose=verbose, **params)
        else:
            res = method(F, grad_F, x, verbose=verbose, **params)
        x_new = list(res.x)
        n_inner += res.n_iter
        fx = f(*x_new)
        n_evals += 1
        log(f"x* = {x_new}, F(x*) = {res.f_x:.8f}, f(x*) = {fx:.8f}")

        # шаг 4 - вычисление P и проверка условия окончания
        p = P(*x_new)
        # для пункта 4б и для таблицы
        g_eq = [g(*x_new) for g, _, _ in equalities]
        g_ineq = [g(*x_new) for g, _, _ in inequalities]
        max_err = max(
            [0]
            + [abs(v) for v in g_eq]
            + [max(0, v) for v in g_ineq]
        )
        log(f"\nШаг 4. g_eq(x*) = {g_eq}, g_ineq(x*) = {g_ineq}, P(x*) = {p:.8f}")
        history.append({
            "k": k, "x_k": x, "r_k": r,
            "x_k1": x_new, "f_xk1": fx, "F_xk1": res.f_x,
            "g_eq": g_eq, "g_ineq": g_ineq, "P": p, "max_err": max_err,
            "inner_iter": res.n_iter, "inner_evals": res.n_evals,
            "inner_reason": res.extra.get("reason"),
        })
        if p <= eps: # штрафная функция p всегда неотрицательна
            reason = "step4"
            log(f"P = {p:.8f} <= eps = {eps} - алгоритм завершён.")
            break
        if k + 1 >= M:
            reason = "max_iter"
            log(f"Достигнуто предельное число внешних итераций M = {M}.")
            break

        # шаг 4б - увеличиваем параметр штрафа и обновляем начальную точку
        r *= C
        x = x_new
        k += 1
        log(f"P > eps => r{k} = {r}, x{k} = {x}. Возврат к шагу 2.")

    return MultivarResult(
        x=x_new, f_x=fx, n_iter=len(history), n_evals=n_evals, history=history,
        extra={"reason": reason, "P": p, "max_err": max_err,
               "inner_iter": n_inner, "F_evals": n_F},
    )
