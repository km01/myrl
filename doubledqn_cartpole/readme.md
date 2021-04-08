

개선사항:

1. double q learning에서 agent가 synchronous한 방식으로 여러 환경에서 에피소드 샘플을 축적함.
2. double q learning에서 현재 네트워크가 $Y^{\mathrm{DoubleDQN}}_t$ 에 대한 역전파를 받느냐, 받지 않느냐가 논문에서는 명시되어있지 않음. 이 코드는 아래의 수식을 따름.

$Y^{\mathrm{DoubleDQN}}_t = R_{t+1} + \gamma Q(S_{t+1}, \mathrm{argmax}_{a} Q(S_{t+1}, a ; \theta_{t}) ;\theta_{t}^{-})$

$\nabla_{\theta_{t}}Y^{\mathrm{DoubleDQN}}_t = \gamma Q(S_{t+1}, \mathrm{argmax}_{a} Q(S_{t+1}, a ; \theta_{t}) ;\theta_{t}^{-}) \cdot \nabla_{\theta_{t}}\max_{a} Q(S_{t+1}, a ; \theta_{t})$

3. CartPole-v1환경은 500번째에서 자동으로 terminal이 되어서 모호한 데이터를 생성하므로, 500번째 데이터는 버린다.



code reference:

http://github.com/higgsfield/RL-Adventure/

http://github.com/g6ling/Reinforcement-Learning-Pytorch-Cartpole/

