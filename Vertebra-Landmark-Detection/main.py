import argparse
import sys
import train
import test
import eval

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=1024, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=14, help='maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_last.pth', help='weights to be resumed')
    parser.add_argument('--weights_dir', type=str, default=None, help='directory for saving/loading weights')
    parser.add_argument('--data_dir', type=str, default='../../Datasets/spinal/', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    parser.add_argument('--max_points', type=int, default=56, help='maximum landmarks used per image (must be multiple of 4)')
    parser.add_argument('--output_dir', type=str, default='outputs/inference_vld', help='base output directory')
    parser.add_argument('--hospital_name', type=str, default='RUIJIN', help='hospital subdirectory name for outputs')
    parser.add_argument('--resolution_csv', type=str, default='', help='Optional CSV mapping sample name to spacing in mm/pixel')
    parser.add_argument('--default_resolution', type=float, default=1.0, help='Fallback spacing when no CSV entry is found')
    parser.add_argument('--matching_mode', type=str, default='hungarian', choices=['hungarian'],
                        help='Landmark matching mode for evaluation metrics (fixed to hungarian)')
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation (horizontal flip)')
    parser.add_argument('--upsample', type=int, default=1, help='Heatmap/reg/wh upsample factor at inference (integer)')
    parser.add_argument('--eval_phase', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to use for eval/test visualization')
    parser.add_argument('--dump_sample_csv', type=str, default='', help='Optional CSV path for per-sample evaluation errors')
    parser.add_argument('--sample_weight_csv', type=str, default='', help='Optional CSV path for per-sample weights used in training')
    parser.add_argument('--max_train_batches', type=int, default=0, help='Limit number of train batches per epoch for smoke tests')
    parser.add_argument('--max_val_batches', type=int, default=0, help='Limit number of val batches per epoch for smoke tests')
    args = parser.parse_args()
    if args.dataset.lower() == 'ruijin' and args.max_points != 52:
        # Ruijin conversion currently provides 52 landmarks per image.
        args.max_points = 52
        if '--max_points' in sys.argv:
            print('Warning: overriding --max_points to 52 for dataset=ruijin')
    if args.max_points % 4 != 0:
        raise ValueError('max_points must be divisible by 4, e.g. 56')
    args.num_vertebra = args.max_points // 4
    args.K = args.num_vertebra
    return args



if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'train':
        is_object = train.Network(args)
        is_object.train_network(args)
    elif args.phase == 'test':
        is_object = test.Network(args)
        is_object.test(args, save=False)
    elif args.phase == 'eval':
        is_object = eval.Network(args)
        is_object.eval(args, save=False)
        # is_object.eval_three_angles(args, save=False)