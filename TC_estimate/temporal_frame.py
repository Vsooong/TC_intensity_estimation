




def get_pretrained_model(load_states=False):
    if load_states:
        encoder = get_resnet('resnet18', False)
    else:
        encoder = get_resnet('resnet18', True)
    n_features = encoder.fc.in_features
    predict_model = Res_Est(encoder, n_features).to(args.device)
    if load_states:
        pass
    return predict_model