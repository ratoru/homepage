---
title: The correct way of adding RGB underglow to your keyboard
description: This guide shows you how set up no EEPROM rgb underglow and layer indication for your split keyboard.
tags: ["keyboard"]
publishDate: 2023-08-17
---

Congratulations! You were brave enough to solder LEDs onto your PCB.

We will make sure the LEDs don't deplete your battery or wear down your microcontrollers. QMK's RGB light documentation is confusing and many keyboard enthusiasts use tricks that aren't mentioned. Setting up RGB underglow can be simple, though.

I will show you how to set up no EEPROM rgb underglow, layer indication, and sleep functionality for your split keyboard.

## No EEPROM lighting

The EEPROM is the persistent storage of your microcontroller and will remember the last settings between reboots and restarts as long as it is not explicitly cleared. EEPROM storage has a limited number of erase cycles, although that number is very high.

I want all my lighting logic to be stored in the RAM of the microcontroller and avoid EEPROM because:

- I want to change the default colors of my keyboard without having to wipe the EEPROM.
- Layer indication would cause many unnecessary writes to the EEPROM.
- If I don't need to waste EEPROM cycles, I don't want to.

## Setup

:::note
In case you have old RGB underglow settings, make sure you wipe the EEPROM using a [designated keycode](https://docs.qmk.fm/#/keycodes?id=quantum-keycodes) or [Bootmagic Lite](https://docs.qmk.fm/#/feature_bootmagic?id=bootmagic-lite).
:::

This guide is for RGB lighting (= underglow). This feature is different (and requires seperate setup) from RGB Matrix (= per-key lighting). You can see my [full keymap on GitHub](https://github.com/ratoru/qmk_keymap).

Start by adding the following to your `rules.mk`. The last two lines were necessary for my keymap to work properly, but depend on your exact setup.

```c
// file: rules.mk
RGBLIGHT_ENABLE = yes
RGB_MATRIX_ENABLE = no  // add this if you are not using per-key lighting.
WS2812_DRIVER = vendor  // add this if you are using a RP2040-based controller.
```

Throughout this guide, I will use `#ifdef RGBLIGHT_ENABLE` to make our keymap compilable with and without RGB light support. Let's add the general configuration to `config.h`:

```c
// file: config.h
#ifdef RGBLIGHT_ENABLE
#define RGB_SLEEP               // Turns off keyboard once host goes to sleep.
#define RGBLIGHT_LIMIT_VAL 128  // Limits max brightness to save energy.
// Add below if WS2812 driver was defined.
#define WS2812_PIO_USE_PIO1 // Force the usage of PIO1 peripheral, by default the WS2812 implementation uses the PIO0 peripheral
#endif
```

`RGBLIGHT_LIMIT_VAL` is the most important setting to reduce energy consumption by limiting the max brightness of the LEDs. Depending on which keyboard you use to compile, you might have to add [other definitions](https://docs.qmk.fm/#/feature_rgblight?id=configuration) like `RGBLIGHT_SPLIT`.

## Basic Underglow

Let's start by simply enabling the LEDs without saving any settings to the EEPROM. All of the following code snippets will be added to `keymap.c` and wrapped with `#ifdef RGBLIGHT_ENABLE`.

```c
// file: keymap.c
void keyboard_post_init_user(void) {
    rgblight_enable_noeeprom(); // Enables RGB, without saving settings.
    rgblight_sethsv_noeeprom(HSV_PINK);
    rgblight_mode_noeeprom(1);  // Animation type. 1 = static.
}
```

## Layer Indication

You can change the underglow color of your keyboard depending on what layer you are on. This can also be done without using the EEPROM, even though the QMK docs do not mention it. Add the following function below the `keyboard_post_init_user` function above. If you have the layer names defined, you can replace `1, 2, 3, ...` with the layer names. They are just `enum`s after all.

```c
// file: keymap.c
layer_state_t layer_state_set_user(layer_state_t state) {
    switch (get_highest_layer(state)) {
     case 1:
         rgblight_sethsv_noeeprom (HSV_MAGENTA);
         break;
     case 2:
         rgblight_sethsv_noeeprom (HSV_BLUE);
         break;
     case 3:
         rgblight_sethsv_noeeprom (HSV_GOLD);
         break;
     case 4:
         rgblight_sethsv_noeeprom (HSV_GREEN);
         break;
     default: //  for any other layers, or the default layer
         rgblight_sethsv_noeeprom (HSV_PINK);
         break;
     }
  return state;
 };
```

## Sleepy LEDs

If your computer goes to sleep, you probably want to save some energy and turn off the keyboard's RGB underglow, too. Since we are using `_noeeprom()` functions, we need to manually do that. Add this below the `layer_state_set_user` function.

```c
// file: keymap.c
void suspend_power_down_user(void) {
    // code will run multiple times while keyboard is suspended
    rgblight_disable_noeeprom();
}

void suspend_wakeup_init_user(void) {
    // code will run on keyboard wakeup
    rgblight_enable_noeeprom();
    rgblight_sethsv_noeeprom(HSV_PINK);
    rgblight_mode_noeeprom(1);
}
```

## Custom RGB Keycodes

The RGB keycodes in the QMK docs use EEPROM settings to affect the LEDs. You can write your own RGB keycodes by defining custom keycodes and calling the corresponding `_noeeprom()` function when the keycode gets processed. Here is an example:

```c
// file: keymap.c
bool process_record_user(uint16_t keycode, keyrecord_t* record) {
  switch (keycode) {
    #ifdef RGBLIGHT_ENABLE
    case RGBT_NE:
      if (record->event.pressed) {
        rgblight_toggle_noeeprom();
      }
      return false;
    #endif
  }
  return true;
}
```

That's it! If you're RGB code is still not working, I recommend asking on Discord or checking out [my implementation](https://github.com/ratoru/qmk_keymap).
