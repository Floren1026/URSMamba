# Net
# ssdf
# CUDA_VISIBLE_DEVICES=0 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-01_19:54:17_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'checkpointH_500.pt' --Rmodel 'checkpointR_500.pt'
# CUDA_VISIBLE_DEVICES=1 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-01_19:54:17_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'best_checkpointH.pt' --Rmodel 'best_checkpointR.pt'

# sum
# CUDA_VISIBLE_DEVICES=1 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-05_12:06:52_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'checkpointH_500.pt' --Rmodel 'checkpointR_500.pt'

# spatial
# CUDA_VISIBLE_DEVICES=1 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-09_16:18:23_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'checkpointH_500.pt' --Rmodel 'checkpointR_500.pt'

#spectral
# CUDA_VISIBLE_DEVICES=1 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-10_09:59:10_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'checkpointH_500.pt' --Rmodel 'checkpointR_500.pt'


# ssdf lhfremamba sum
# CUDA_VISIBLE_DEVICES=0 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-12_16:54:02_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'checkpointH_500.pt' --Rmodel 'checkpointR_500.pt'


# ssdf no lhfremamba sum
# CUDA_VISIBLE_DEVICES=0 python3 test_multi.py --imageSize 256 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-14_17:07:26_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'checkpointH_500.pt' --Rmodel 'checkpointR_500.pt'



# CUDA_VISIBLE_DEVICES=1 python3 test_multi.py --imageSize 256 --channel_secret 3 --batch_stegs 1 --num_secret 1 --Model_dir '2025-07-08_22:55:43_144_0.001_0.001_0.75_l2_3In3' --Hmodel 'best_checkpointH.pt' --Rmodel 'best_checkpointR.pt'

CUDA_VISIBLE_DEVICES=1 python3 test_multi.py --imageSize 256 --channel_secret 8 --batch_stegs 1 --num_secret 1 --Model_dir '2025-06-01_19:54:17_144_0.001_0.001_0.75_l2_8In3' --Hmodel 'best_checkpointH.pt' --Rmodel 'best_checkpointR.pt'