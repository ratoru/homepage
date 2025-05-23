---
title: "Building a custom keyboard layout for ergonomic kbds (1/3)"
description: "Learn how to build an optimized keyboard layout. Part 1/3 covering how to choose a base layout that fits your needs."
tags: ["coding", "qmk"]
publishDate: 2023-07-23
---

![kyria-heatmap](../../assets/blog/layout/kyria-heatmap.webp)

I have healthier wrists and am a more efficient typer because I designed my own keyboard layout.

This guide will take you from having never heard about what a keyboard layout is all the way to using your custom layout on a mechanical keyboard via `QMK`. Even though the ergonomic keyboard community is very kind and helpful, most of the information is buried across Reddit, old blogs, and even older internet forums. I will walk you through all the necessary information a complete beginner needs to start building a custom layout.

Let's begin our journey down the massive rabbit hole that is keyboard layouts.

## The Basics

You just came off your high discovering [ergonomic keyboards](https://ratoru.com/blog/keyboard) and have now realized that you need to put all the keys you just lost somewhere.

Let's take this opportunity to revamp your typing habits and shortcuts to make your typing more efficient. In the long run, good typing posture and shorter finger movements can even help prevent repetitive strain injury (RSI).

Building your custom layout will consist of 3 steps:

1. Chose a keyboard layout for your base layer, for example `Qwerty`, `Dvorak`, or `Colemak-DH`.
2. Customize the layout to match your specific keyboard. This is the fun part since you get to infuse your keyboard with superpowers.
3. Install your custom layout by flashing firmware to your keyboard, for example by using [QMK Firmware](https://qmk.fm/).

Don't worry if these steps seem daunting. Nothing beats experimenting with all the different features and seeing what you like. You will become a better typer along the way.

## Picking a Base Layout

If you are like me, you typed on a `Qwerty` [^1] your entire life; turns out `Qwerty` is garbage in terms of ergonomics.

[^1]: The `qwerty` layout is what you see on any regular keyboard. It's named after the first letters in the upper row.

Certain letters and bigrams appear much more often in text or code than others. Unfortunately, `Qwerty` was not designed to make the most frequent letters and bigrams the most comfortable to hit. Naturally, many keyboard layouts have since been designed that optimize specifically for typing comfort.

The three most popular layouts are `Qwerty`, `Dvorak`, and `Colemak(-DH)`. The images below are heatmaps for these three layouts that were generated using 10000 sentences from Wikipedia. The first time I saw these heatmaps, I was shocked. I recommend you watch [this video](https://youtu.be/gRtS-XACO6o?t=42) to learn a little bit about the history of these (and other) layouts.

![qwerty-heatmap](../../assets/blog/layout/qwerty-heatmap.webp)
![dvorak-heatmap](../../assets/blog/layout/dvorak-heatmap.webp)
![colemak-dh-heatmap](../../assets/blog/layout/colemak-dh-heatmap.webp)

Unfortunately, every English keyboard runs on `Qwerty`, so switching layouts does have drawbacks. Read through Pascal Getreuer's excellent article on [alternate keyboard layouts](https://getreuer.info/posts/keyboards/alt-layouts/index.html)and decide whether you want to learn one. For lazy readers here is my quick summary. The primary motivation for switching layouts is improved typing comfort. Improving speed should not weigh into your decision. Additionally, note that learning a new layout is hard and you will have to use `qwerty` when using someone else's computer. On the other hand, alternate keyboard layouts like `Colemak-DH` and `Sturdy` are significantly more ergonomic than `qwerty`. If you want to switch layouts, but can't decide on a specific one, go with `Colemak-DH`. [Colemak-DH](https://colemakmods.github.io/mod-dh/) has been battle-tested by the community.

> Your fingers on QWERTY move 2.2x more than on Colemak. QWERTY has 16x more same hand row jumping than Colemak. There are 35x more words you can type using only the home row on Colemak.
> -- [colemak.com](https://colemak.com/)

Personally, I am currently exploring switching to Colemak-DH. I also designed a custom layout that let's me use `hjkl` in `vim` to move around. If you'd like to switch, I recommend using [monkeytype.com](https://monkeytype.com/) and [keybr.com](https://www.keybr.com/) to {% footnote idNumber=2 label="practice" %} It is also recommended to change the corpus on `monkeytype` from `🌐 english` to `🌐 english 5k` right above the text, so that you come across more words while practicing. {% /footnote %}. I recommend at least trying a different layout, although you won't get any good without a serious commitment. And here is more text. akcsjaksdcjsc.

## Next steps

Now it is time to customize your layout for your specific keyboard. Aditionally, you need to add symbols, numbers, and all the other things that don't fit on your keybaord. Along the way you will be able to create keys that select entire words or type `../` with one press. I show you how to do it in [part 2 of this guide](https://ratoru.com/blog/layout-customizing).
