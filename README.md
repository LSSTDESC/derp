# derp
A first attempt at a simple LSST DRP catalog emulator

## Contributing

To help develop `derp`, you'll need a clone of this repo in your `$HOME/desc` folder at NERSC:
```bash
cd ~/desc
git clone git@github.com:LSSTDESC/derp.git
```
Then, you'll need to insert this to your python path as you need it, like this:
```python
import os, sys
derp_dir = os.environ['HOME']+'/desc/derp'
sys.path.insert(0, derp_dir)
```
To push to a branch on the base repo you'll need Write access: contact @drphilmarshall about this. Alternatively, feel free to fork the repo and submit PRs from afar.