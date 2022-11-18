from torchattacks import PGD, FGSM, PGDL2, AutoAttack, CW


# get attack 
def get_attack(net, args):
    if args.att_method == 'fgsm':
        attack = FGSM(net, args.epsilon)
        annotation = f'{args.att_method}_eps({args.epsilon:.2f})'
    elif args.att_method == 'pgd':
        if args.att_norm == 'Linf':
            attack = PGD(net, eps=args.epsilon, alpha=args.alpha, steps=args.steps)
            annotation = f'{args.att_method}_eps({args.epsilon:.2f})_alpha({args.alpha:.2f})_steps({args.steps})'

        elif args.att_norm == 'L2':
            attack = PGDL2(net, eps=args.epsilon, alpha=args.alpha, steps=args.steps)
            annotation = f'{args.att_method}_eps({args.epsilon:.2f})_alpha({args.alpha:.2f})_steps({args.steps})'

    elif args.att_method == 'cw':
        attack = CW(net,c=args.c, kappa=args.kappa, steps=args.steps, lr=args.lr)
        annotation = f'{args.att_method}_c({args.c})_kappa({args.kappa})_lr({args.lr})_steps({args.steps})'

    elif args.att_method == 'aa':
        attack = AutoAttack(net, norm=args.norm, eps=args.epsilon)
        annotation = f'{args.att_method}_eps({args.epsilon:.2f})'

    return annotation, attack


def dist_of(imgs, adv_imgs, Lp_norm):
    '''
    Args: 
    delta_imgs of size (B C H W)
    Lp_norm is the norm used
    '''
    delta_imgs = imgs - adv_imgs
    delta_imgs = delta_imgs.flatten(start_dim=1).abs()
    if Lp_norm == 'Linf':
        dists = delta_imgs.max(dim=1)[0]
    if Lp_norm == 'L2':
        dists = (delta_imgs**2).sum(dim=1).sqrt()    

    return dists



