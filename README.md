# Coach-RL

- `pip install -e .`
- Build your vss-software
- Copy vss-software/src/Config to examples
- Copy your builded agent to gym_coach_vss/bin
- run examples/sample_coach.py

ps. CoachEnv takes as argument "sim_path". This argument indicates where 
your FIRASim binarie is. The default will take it as '/home/$USER/FIRASim/bin/FIRASim'.