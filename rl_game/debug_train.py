import traceback
from training.trainer import Trainer

try:
    trainer = Trainer(episodes=2)
    trainer.run()
except Exception as e:
    print('EXC', type(e).__name__, e)
    traceback.print_exc()
