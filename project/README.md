# EMP Project

## Realtime EMP goal tracking (PyBullet)

Run:

```bash
python run_book_rack_realtime.py
```

This launches a receding-horizon EMP loop that keeps adapting online while the
goal rack pose changes.

Controls:

- Mouse: move `goal_x`, `goal_y`, `goal_z`, `goal_yaw_deg` sliders in the PyBullet GUI
- Arrow keys: move goal in `x/y`
- `U` / `J`: move goal in `z`
- `Q` / `E`: rotate goal yaw
- `R`: reset scene and robot start
- `X`: exit

Useful tuning flags:

```bash
python run_book_rack_realtime.py --adapt-hz 4 --n-vel 70 --plan-steps 80
```
