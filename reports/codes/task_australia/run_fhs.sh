#!/bin/bash
FHS=/nix/store/37d2hrh6q9943gbzlxrnrwbbc3vwfqa3-steam-run-1.0.0.85-fhsenv-rootfs
BWRAP=/nix/store/f2bsh1n6snqdldnjdshdy1lw3hnlfxwi-bubblewrap-0.12.0/bin/bwrap
FONTDIR=/nix/store/fa2vzgw2pds7gjb9671n9fbphfw65sk3-dejavu-fonts-2.37/share/fonts/truetype
exec $BWRAP --dev-bind /dev /dev --proc /proc --bind /nix /nix --bind /tmp /tmp --bind /home /home --bind /run /run \
  --ro-bind $FHS/usr /usr --ro-bind $FHS/etc /etc \
  --symlink /usr/bin /bin --symlink /usr/sbin /sbin --symlink /usr/lib /lib --symlink /usr/lib64 /lib64 \
  --ro-bind $FONTDIR /usr/share/fonts/truetype \
  --chdir "$PWD" --setenv DISPLAY :99 --setenv SDL_AUDIODRIVER dummy --setenv LD_LIBRARY_PATH "/usr/lib64:." \
  -- "$@"
