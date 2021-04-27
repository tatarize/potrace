# Python potrace
Pure Python Port of Potrace. This is a python port of Peter Selinger's Potrace (based on 1.16 code).

<img width="200" height="200" src="https://gist.githubusercontent.com/tatarize/42884e5e99cda88aa5ddc2b0ab280973/raw/488cafa1811bd2227458390804910fbc4a90b9ea/head.svg"/>

![head-orig3](https://user-images.githubusercontent.com/3302478/115929160-2757f180-a43c-11eb-88dc-1320706c9a3f.png)

This port is needed because many python hooks to the original code such as `pypotrace` have installation issues and compilation problems with some OSes. This potrace is written in pure python and will be compatible with basically anything.

# Installing

To install or use as a requirement:
* `pip install potrace`

### Potrace-CLI
If you wish to use the Command Line Interface that is stored in a sister project `potrace-cli` (https://github.com/tatarize/potrace-cli). This can be installed with:
* `pip install potrace[cli]`

or:

* `pip install potrace-cli`

The cli project contains valid console script entrypoints for potrace. If you install the command-line package it will add `potrace` to your console scripts.

Warning: This may conflict with runtime namespace of regular potrace. Be certain this is your intent. `pip uninstall potrace-cli` will uninstall if needed.


# Requirements
* numpy: for bitmap structures.

# Parallel Projects
This project intentionally duplicates a considerable amount of the API of `pypotrace` such that this library can be used as a drop-in replacement.

This library offers CLI potrace as an optional package, to permit performing potrace commands from the commandline.

# License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2, or (at your option) any later version.

Furthermore, this is permitted to be relicensed under any terms the Peter Selinger's original Potrace is licensed under. If he broadly publishes the software under a more permissive license this port should be considered licensed as such as well. Further, if you purchase a proprietary license for inclusion within commercial software under his Dual Licensing program your use of this software shall be under whatever terms he permits for that. Any contributions to this port must be made under equally permissive terms.

"Potrace" is a trademark of Peter Selinger. Permission granted by Peter Selinger.
