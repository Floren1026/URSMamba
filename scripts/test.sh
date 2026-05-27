# CUDA_VISIBLE_DEVICES=0 python3 test.py --imageSize 256 --batch_stegs 1 --num_training 1 --num_secret 2 --Model_dir '2025-02-05_13:35:13_144_0.001_0.001_0.75_0.75_0.75_l2_2In1_lab20231-System-Product-Name' --Hmodel 'checkpointH_040.pt' --Rmodel 'checkpointR_040.pt'

CUDA_VISIBLE_DEVICES=0 python3 test.py --imageSize 256 --batch_stegs 1 --num_training 1 --num_secret 1 --Model_dir '2025-07-20_10:26:34_144_0.001_0.001_0.75_l2_3In3' --Hmodel 'best_checkpointH.pt' --Rmodel 'best_checkpointR.pt'
