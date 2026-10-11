from train import DEC_Module
###################################################################
# Settings
cfg = {'dec_mode': True, 'train': True, 'test':False, 'evals':False, 'vis': False}

resume = None

if cfg['dec_mode']:
    if cfg['train']:
        dec_model = DEC_Module(multigpu=False, resume=resume)
        dec_model.train(vis=cfg['vis'])
    if cfg['test']:
        dec_model = DEC_Module(multigpu=False, resume=resume)
        dec_model.test()
    if cfg['evals']:
        dec_model = DEC_Module(multigpu=False, resume=resume)
        dec_model.eval_single()

"""
DIOU: train: VOC07 trainval + VOC12 trainval
        test: VOC07 test
        Mean AP = 0.7351

CIOU: train: VOC07 trainval + VOC12 trainval
        test: VOC07 test
        Mean AP = 

GIOU: train: VOC07 trainval + VOC12 trainval
        test: VOC07 test
        Mean AP = 
"""