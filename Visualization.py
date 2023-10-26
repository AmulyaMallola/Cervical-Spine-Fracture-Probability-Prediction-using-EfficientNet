import matplotlib.pyplot as plt
import numpy as np
best_val_loss = float('inf')
best_model_path = 'best_modell.h5'

for train_idx, val_idx in StratifiedKFold(2).split(train_df, train_df['patient_overall']):
    K.clear_session()
    x_train = train_df.iloc[train_idx].reset_index()
    x_val = train_df.iloc[val_idx].reset_index()
    model = get_model()
    hist = model.fit(
        TrainGenerator(x_train, min(len(x_train), 64), base_path=train_dir),
        steps_per_epoch=max((len(x_train) // 64), 1),
        epochs=50,
        validation_data=TrainGenerator(x_val, min(len(x_val), 64), base_path=train_dir),
        validation_steps=max((len(x_val) // 64), 1)
    )

    # Save the best model based on validation loss
    if hist.history['val_loss'][0] < best_val_loss:
        best_val_loss = hist.history['val_loss'][0]
        model.save(best_model_path)
        print("Saved the best model at epoch", len(hist.history['val_loss']))

    val_gen = TrainGenerator(x_val, min(len(x_val), 64), base_path=train_dir)

    # Continue with the remaining code...


    try: 
        preds = model.predict(TestGenerator(test_df, min(len(test_df), 64), infinite = False, base_path = test_dir), steps = max((len(test_df) // 64), 1))
        
        new_preds = []
        for pred_idx in range(len(preds)):
            new_preds.append(preds[pred_idx][prediction_type_mapping[pred_idx]])
        
        submission['fractured'] += np.array(new_preds) / 5
        
    except: traceback.print_exc()