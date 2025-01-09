import bson

def recover_model(filename):
    opt = None
    # Load all the BSON files
    with open(f"{filename}_hps.bson", 'rb') as f:
        hps = bson.loads(f.read())
    with open(f"{filename}_progress.bson", 'rb') as f:
        store = bson.loads(f.read())    
    with open(f"{filename}_mod.bson", 'rb') as f:
        network = bson.loads(f.read())    
    with open(f"{filename}_policy.bson", 'rb') as f:
        policy = bson.loads(f.read())    
    with open(f"{filename}_prediction.bson", 'rb') as f:
        prediction = bson.loads(f.read())
    return network, opt, store, hps, policy, prediction

def save_model(m, store, opt, filename, environment, loss_hp, Lplan):
    model_properties = m.model_properties
    network = m.network   
    # Prepare hps dictionary
    hps = {
        "Nhidden": model_properties['Nhidden'],
        "T": environment['dimensions']['T'],
        "Larena": environment['dimensions']['Larena'],
        "Nin": model_properties['Nin'],
        "Nout": model_properties['Nout'],
        "GRUind": m.GRUind,  # Assuming GRUind is a part of the model
        "βp": loss_hp['βp'],
        "βe": loss_hp['βe'],
        "βr": loss_hp['βr'],
        "Lplan": Lplan,
    }

    # Save the data using bson
    with open(filename + "_progress.bson", 'wb') as f:
        f.write(bson.BSON.encode({'store': store}))
    with open(filename + "_mod.bson", 'wb') as f:
        f.write(bson.BSON.encode({'network': network}))
    with open(filename + "_opt.bson", 'wb') as f:
        f.write(bson.BSON.encode({'opt': opt}))
    with open(filename + "_hps.bson", 'wb') as f:
        f.write(bson.BSON.encode({'hps': hps}))

    if hasattr(m, 'policy'):
        policy = m.policy
        with open(filename + "_policy.bson", 'wb') as f:
            f.write(bson.BSON.encode({'policy': policy}))
    if hasattr(m, 'prediction'):
        prediction = m.prediction
        with open(filename + "_prediction.bson", 'wb') as f:
            f.write(bson.BSON.encode({'prediction': prediction}))
    if hasattr(m, 'prediction_state'):
        prediction_state = m.prediction_state
        with open(filename + "_prediction_state.bson", 'wb') as f:
            f.write(bson.BSON.encode({'prediction_state': prediction_state}))

