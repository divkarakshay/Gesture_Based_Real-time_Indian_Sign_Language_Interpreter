python3 "video-to-frame.py" train_videos train_frames
python3 "video-to-frame.py" test_videos test_frames
python retrain.py --bottleneck_dir=bottlenecks --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=train_frames
python predict_spatial.py retrained_graph.pb train_frames --batch=100
python predict_spatial.py retrained_graph.pb test_frames --batch=100 --test
python predict_spatial.py retrained_graph.pb train_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100
python predict_spatial.py retrained_graph.pb test_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100 --test
python rnn_train.py predicted-frames-final_result-train.pkl non_pool.model
y
python rnn_train.py predicted-frames-GlobalPool-train.pkl pool.model
y
python rnn_eval.py predicted-frames-final_result-test.pkl non_pool.model
python rnn_eval.py predicted-frames-GlobalPool-test.pkl pool.model
pause
