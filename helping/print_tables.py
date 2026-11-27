def print_summary_table(history):
    header = (f"{'k':>2} | {'[a_k; b_k]':^20} | {'y_k':>8} | {'f(y_k)':>8} | "
              f"{'z_k':>8} | {'f(z_k)':>8} | {'Случай':^6} | {'Новый интервал':^20} | "
              f"{'Delta':>7}")
    print(header)
    for row in history:
        interval = f"[{row['a_k']:.4f}; {row['b_k']:.4f}]"
        new_interval = f"[{row['a_k1']:.4f}; {row['b_k1']:.4f}]"
        print(f"{row['k']:>2} | {interval:^20} | {row['y_k']:>8.4f} | {row['f_yk']:>8.4f} | "
              f"{row['z_k']:>8.4f} | {row['f_zk']:>8.4f} | {row['case']:^6} | "
              f"{new_interval:^20} | {row['delta']:>7.4f}")

def print_table_multivar(history):
    marq = bool(history) and "mu_k" in history[0]
    dfp = bool(history) and "A_k" in history[0]
    head = [f"{'k':>2}", f"{'x^k':^22}", f"{'f(x^k)':>9}", f"{'||grad||':>9}"]
    if marq:
        head += [f"{'mu_k':>9}", f"{'Попыток':>7}", f"{'d^k':^22}", f"{'x^(k+1)':^22}", f"{'f(x^(k+1))':>10}"]
    elif dfp:
        head += [f"{'A^k':^44}", f"{'d^k':^22}", f"{'t_k':>8}", f"{'x^(k+1)':^22}",
                 f"{'||norm_x||':>8}", f"{'|mod_fx|':>8}", f"{'Условия':^7}"]
    else:
        head += [f"{'d^k':^8}", f"{'t_k':>8}", f"{'x^(k+1)':^22}", f"{'||norm_x||':>8}",
                 f"{'|mod_fx|':>8}", f"{'Условия':^7}"]
    print(" | ".join(head))
    for row in history:
        xk = f"({row['x_k'][0]:.4f}; {row['x_k'][1]:.4f})"
        xk1 = f"({row['x_k1'][0]:.4f}; {row['x_k1'][1]:.4f})"
        cells = [f"{row['k']:>2}", f"{xk:^22}", f"{row['f_xk']:>9.4f}", f"{row['grad_norm']:>9.4f}"]
        if marq:
            dk = f"({row['d'][0]:.4f}; {row['d'][1]:.4f})"
            cells += [f"{row['mu_k']:>9.4f}", f"{row['trials']:>7}", f"{dk:^22}", f"{xk1:^22}",
                      f"{row['f_xk1']:>10.4f}"]
        elif dfp:
            ok = "да" if row["ok"] else "нет"
            A = row["A_k"]
            Ak = f"(({A[0][0]:.4f}; {A[0][1]:.4f}); ({A[1][0]:.4f}; {A[1][1]:.4f}))"
            dk = f"({row['d'][0]:.4f}; {row['d'][1]:.4f})"
            cells += [f"{Ak:^44}", f"{dk:^22}", f"{row['t_k']:>8.4f}", f"{xk1:^22}",
                      f"{row['norm_x']:>8.4f}", f"{row['mod_fx']:>8.4f}", f"{ok:^7}"]
        else:
            ok = "да" if row["ok"] else "нет"
            cells += [f"{row.get('direction', ''):^8}", f"{row['t_k']:>8.4f}", f"{xk1:^22}",
                      f"{row['norm_x']:>8.4f}", f"{row['mod_fx']:>8.4f}", f"{ok:^7}"]
        print(" | ".join(cells))

def print_table_constrained(history):
    # для функции двух переменных с одним ограничением-равенством
    lagrange = bool(history) and "lambda_k" in history[0]
    func_name = "L" if lagrange else "F"

    head = [f"{'k':>2}", f"{'r_k':>8}"]

    if lagrange:
        head += [f"{'lambda_k':>12}"]

    head += [f"{'x^k':^24}", f"{'x*':^24}",
             f"{'f(x*)':>11}", f"{func_name + '(x*)':>11}",
             f"{'g(x*)':>12}", f"{'P(x*)':>12}",
             f"{'Внутр. ит.':>10}", f"{'Выч. ' + func_name:>8}"]

    print(" | ".join(head))

    for row in history:
        xk = f"({row['x_k'][0]:.6f}; {row['x_k'][1]:.6f})"
        xk1 = f"({row['x_k1'][0]:.6f}; {row['x_k1'][1]:.6f})"

        cells = [f"{row['k']:>2}", f"{row['r_k']:>8.2g}"]

        if lagrange:
            cells += [f"{row['lambda_k'][0]:>12.6f}"]

        cells += [f"{xk:^24}", f"{xk1:^24}",
                  f"{row['f_xk1']:>11.6f}",
                  f"{row[func_name + '_xk1']:>11.6f}",
                  f"{row['g_eq'][0]:>12.3e}",
                  f"{row['P']:>12.3e}",
                  f"{row['inner_iter']:>10}",
                  f"{row['inner_evals']:>8}"]

        print(" | ".join(cells))
