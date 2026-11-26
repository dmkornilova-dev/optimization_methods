from helping.log import make_logger
from helping.types import MultivarResult


def lagrange_method(f, grad, hess, x0, equalities, inequalities,
                    r0, C, lambda0, mu0, eps, min_method, M=100, verbose=False):
    # каждое ограничение задаётся тройкой (g, grad_g, hess_g)
    # equalities: g(x) = 0; inequalities: g(x) <= 0
    if r0 <= 0 or C <= 1 or eps <= 0 or M <= 0:
        raise ValueError("Требуется r0 > 0, C_LAGR > 1, eps > 0, M > 0")
    if len(lambda0) != len(equalities) or len(mu0) != len(inequalities):
        raise ValueError("Число множителей должно совпадать с числом ограничений")
    if any(v < 0 for v in mu0):
        raise ValueError("Начальные множители mu должны быть неотрицательными")

    log = make_logger(verbose)
    params = dict(min_method)
    method = params.pop("func")
    use_hess = params.pop("use_hess", False) #требуется ли выбранному методу матрица Гессе

    # шаг 1
    x = list(x0)
    r, lam, mu = r0, list(lambda0), list(mu0) # параметр штрафа, вектора множителей
    k = 0
    history = []
    n_evals = 0 # количество вычислений целевой функции
    n_inner = 0 # суммарное количество внутренних итераций
    n_L = 0 # количество вычислений функции Лагранжа
    log("Метод множителей Лагранжа - подробное решение\n")
    log(f"Шаг 1. x0 = {x}, r0 = {r}, C_LAGR = {C}, lambda0 = {lam}, mu0 = {mu}, eps = {eps}, k = 0")

    while True:
        # шаг 2 - модифицированная функция Лагранжа L = f + lambda*g + P
        def P(*point): # функция штрафа
            p_eq = r / 2 * sum(g(*point)**2 for g, _, _ in equalities)
            p_ineq = sum(max(0, mu_j + r*g(*point))**2 - mu_j**2
                         for mu_j, (g, _, _) in zip(mu, inequalities)) / (2*r) # равна нулю при отсутствии ограничений-неравенств
            return p_eq + p_ineq

        def L(*point): # модифицированная функция Лагранжа
            nonlocal n_evals, n_L
            n_evals += 1
            n_L += 1
            return (f(*point) + sum(v*g(*point) for v, (g, _, _) in zip(lam, equalities))
                    + P(*point))

        def grad_L(*point): # для методов лабы 4: градиент функции, которую минимизируем
            # ищем по правилам взятия производных
            result = list(grad(*point)) # сначала градиент исходной функции
            for lam_j, (g, gg, _) in zip(lam, equalities):
                coef = lam_j + r*g(*point)
                result = [a + coef*b for a, b in zip(result, gg(*point))] # прибавили (lam_j+r*g(x))*grag_g(x)
            for mu_j, (g, gg, _) in zip(mu, inequalities):
                coef = max(0, mu_j + r*g(*point))
                result = [a + coef*b for a, b in zip(result, gg(*point))]
            return result

        def hess_L(*point): # hess_L = hess(f) + sum (r * grad(g) * grad(g)^T + (lambda_j + r*g) * H(g))
            result = [list(row) for row in hess(*point)] # первое слагаемое - гессиан целевой функции
            # ограничения-равенства
            for lam_j, (g, gg, hg) in zip(lam, equalities):
                coef = lam_j + r * g(*point) # скобка для второй части слагаемого
                dg = gg(*point) # градиент ограничения для первой части слагаемого
                H = hg(*point) # гессиан ограничения для второй части слагаемого
                for i in range(len(point)):
                    for j in range(len(point)):
                        result[i][j] += r * dg[i] * dg[j] # r * grad(g) * grad(g)^T - первая часть слагаемого
                        result[i][j] += coef * H[i][j] # (lambda_j + r*g) * H(g) - вторая часть слагаемого

            # ограничения-неравенства
            for mu_j, (g, gg, hg) in zip(mu, inequalities):
                coef = mu_j + r * g(*point)
                # обе поправки учитываются только при mu_j + r*g > 0
                if coef > 0:
                    dg = gg(*point)
                    H = hg(*point)
                    for i in range(len(point)):
                        for j in range(len(point)):
                            result[i][j] += r * dg[i] * dg[j]
                            result[i][j] += coef * H[i][j]

            return result

        log(f"\nШаг 2 (итерация k = {k}). x{k} = {x}, r{k} = {r}, lambda{k} = {lam}, mu{k} = {mu}")
        log("L(x) = f(x) + sum(lambda_j*g_j(x)) + P(x)")
        if verbose: # вывод точной формулы модифицированной функции Лагранжа на экран
            terms = ["f(x)"]
            for j, lambda_j in enumerate(lam, 1):
                terms.append(f"({lambda_j})*g{j}(x)")
                terms.append(f"{r / 2}*g{j}(x)^2")
            for j, mu_j in enumerate(mu, len(equalities) + 1):
                terms.append(
                    f"1/{2 * r} * (max(0, {mu_j} + {r}*g{j}(x))^2 - ({mu_j})^2)"
                )
            log("L(x) = " + " + ".join(terms))

        # шаг 3 - безусловная минимизация L методом из лабы 4
        log(f"\nШаг 3. Минимизируем L методом {method.__name__}, начальная точка: {x}")
        if use_hess:
            res = method(L, grad_L, hess_L, x, verbose=verbose, **params)
        else:
            res = method(L, grad_L, x, verbose=verbose, **params)
        x_new = list(res.x)
        n_inner += res.n_iter
        fx = f(*x_new)
        n_evals += 1
        log(f"x* = {x_new}, L(x*) = {res.f_x:.8f}, f(x*) = {fx:.8f}")

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
            "k": k, "x_k": x, "r_k": r, "lambda_k": lam[:], "mu_k": mu[:],
            "x_k1": x_new, "f_xk1": fx, "L_xk1": res.f_x,
            "g_eq": g_eq, "g_ineq": g_ineq, "P": p, "max_err": max_err,
            "inner_iter": res.n_iter, "inner_evals": res.n_evals,
            "inner_reason": res.extra.get("reason"),
        })
        if abs(p) <= eps:
            reason = "step4"
            log(f"|P| = {abs(p):.8f} <= eps = {eps} - алгоритм завершён.")
            break
        if k + 1 >= M:
            reason = "max_iter"
            log(f"Достигнуто предельное число внешних итераций M = {M}.")
            break

        # шаг 4б - сначала множители пересчитываются со старым значением r_k
        lam = [lam_j + r*g for lam_j, g in zip(lam, g_eq)]
        mu = [max(0, mu_j + r*g) for mu_j, g in zip(mu, g_ineq)]
        # а потом находим новый r
        r *= C
        x = x_new
        k += 1
        log(f"|P| > eps => r{k} = {r}, lambda{k} = {lam}, mu{k} = {mu}, x{k} = {x}. Возврат к шагу 2.")

    return MultivarResult(
        x=x_new, f_x=fx, n_iter=len(history), n_evals=n_evals, history=history,
        extra={"reason": reason, "P": p, "max_err": max_err,
               "inner_iter": n_inner, "L_evals": n_L},
    )
