import attack_adaptor as attacks
defenders_name = ['plain_dnn','FGSM_train', 'PGD_train', 'distillation', 'quant','reg_dnn', 'IBP_train']
attackers_name = ['noise', 'SPSA', 'UAP_0.03', 'FGSM', 'PGD', 'CWinf', 'DeepFool', 'Milp']
for d_name in defenders_name:
    print('*************')
    print(d_name)
    names = {'dataset' : 'TEP', 'model' : d_name}
    clean = attacks.CleanAdaptor(**names)
    cln_v = clean.verify()
    print('{:.2%}'.format(cln_v))
    for a_name in attackers_name:
        print(a_name)
        atk = attacks.AttackAdaptor(**names,attack_name=a_name)
        acc, radius, eps_inf, epsilon_crr, _ = atk.get_adv_info(0.03, False)
        print('acc:{:.2%} radius:{:.3}'.format(acc,radius))