---
title: "Creating and flashing your keyboard firmware using QMK (3/3)"
description: "Learn how to build an optimized keyboard layout. Part 3/3 covering how to actually create your keyboard layout and flash your keyboard with it."
tags: ["keyboard"]
publishDate: 2023-07-25
---

This is where we turn layout dreams into reality.

I will show you how to put together all the files you need to get your custom layout on to your keyboard. Using GitHub Actions we can avoid the trouble of installing `QMK` locally to compile our firmware. Lastly, we will cover how to flash your keyboard.

Your hands will thank you for the beautiful keyboard layout you created for them.

:::note
This is part 3/3 of my cutom keyboard layout series. I recommend you read [part 2](https://ratoru.com/blog/layout-customizing) before continuing.
:::

## Create the necessary files

If you have no programming experience, I recommend using [Via](https://www.caniusevia.com/) to set up your keyboard. If you know how to use `git` and want a little extra control over your keymap, you should use `QMK` directly. The rest of this guide will cover how to use `QMK` directly to write custom macros.

Use [QMK Configurator](https://config.qmk.fm/#/) to create your layers and most functionality. Here is a [good tutorial](https://www.youtube.com/watch?v=-imgglzDMdY). Once you have a `json` file with your basic layers, let's add all the macros that you might need. These macros will require additional setup, unfortunately.

The `qmk_firmware` GitHub repo is a little unwieldy. I think there are two main ways of installing QMK and using it effectively.

- [Using GitHub Actions](https://docs.qmk.fm/#/newbs_building_firmware_workflow). This is my preferred method since you can avoid installing `qmk`.
- [Using a git submodule](https://medium.com/@patrick.elmquist/separate-keymap-repo-for-qmk-136ff5a419bd). This requires the [QMK CLI](https://docs.qmk.fm/#/newbs_getting_started?id=installation-1) to be installed.

Lastly, I recommend you decrease the size of your firmware by following [this guide](https://docs.qmk.fm/#/squeezing_avr).

## Compile using GitHub Actions

This compilaton method lets you avoid installing `qmk` on your local machine. You can see it at work in my [qmk repo](https://github.com/ratoru/qmk_userspace/tree/main). The QMK docs cover setting up the repository, well. Unfortunately, there is barely any information on how to customize it properly. If you get stuck, I recommend asking in the QMK Discord. In July 2023 you had to do the following steps to get custom macros to work with GitHub Actions:

- Place `process_record_user` in `source.c`.
- Add `SRC += source.c` to `rules.mk`.
- Define custom keycodes in `config.h` inside `ifndef __ASSEMBLER__`.
- Enable any QMK features you want in `rules.mk`.
- If you have a Litaris controller, add `-e CONVERT_TO=liatris` to the `build.yml` compile command.

## Flashing your keyboard

:::note
Remember that you can always ask for help in your keyboard's Discord.
:::

This depends on the type of controller you bought. If your controller uses `uf2` files, you can avoid `QMK Toolbox` or the `QMK CLI` all together. Follow [these steps](https://docs.splitkb.com/hc/en-us/articles/6330981035676-Aurora-Build-Guide-20-Flashing-Firmware) for the flashing process.

If you have to use `QMK Toolbox` or the command line to flash your keyboard, follow the [official docs](https://docs.qmk.fm/#/newbs_flashing).

Your keyboard is good to go!
