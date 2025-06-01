---
title: Reflecting on two years of using split keyboards
description: Personal learnings from using multiple split keyboards and layouts for two years. I hope you can avoid some of my mistakes.
tags: ["keyboard"]
publishDate: 2025-06-01
---

My ergonomic keyboard journey over the last two years was very inefficient.

I have overspent buying different keyboards, switches, and keycaps. I've invested countless hours optimizing my layout. And worst of all, I spent a year learning `Colemak-DH` despite there being better options available.

Here are some of my personal opinions regarding custom keyboards. I hope you can avoid some of my mistakes.

## Things I wish I knew sooner

- Colemak-DH is an outdated layout. There are newer, better keyboard layouts available like Graphite or Gallium. Take a look at my [layout article](https://ratoru.com/blog/choose-the-right-base-layout/#the-stats) for more.
- MX switches are nicer to type on, but make the keyboard significantly taller. If you have to bend your wrists while typing, you are throwing away all the ergonomics.
- Invest in a switch sampler. It took me a while and a lot of money to find the kind of switches I like. I'm currently using silent, linear switches. I usually try to go for lighter switches.
- Switching between Qwerty and a different layout has been no problem at all. I keep them separated in my mind by only using Gallium on my split and only Qwerty on my laptop. This has probably slowed down my learning, but it doesn't bother me.
- The RP2040 based controllers are a lot better than the Pro Micros. I would heavily favor boards that can use these modern controllers and have extra storage. You will be able to have more RGB lighting and Oled effects. Additionally, the flashing process much simpler.
- I hate the input delay of home row mods. A key with a tap-hold decision fires on key up instead of key down, and it trips me up! I will definitely try again as people keep writing better implementations of home row mods and my typing speed increases. For now I use Callum-style one shot mods and combos. For a modern implementation, check out my [qmk-userspace](https://github.com/ratoru/qmk_userspace/tree/main).
- Soldering is likely not worth the effort. There are really good keyboards out there that come pre-soldered like the Halycon series. If you don't already have soldering equipment, the extra price is likely worth it. Soldering a keyboard takes roughly 2 full days for me. This was a fun experience the first few times, but from now on I would like to save that time.
- Expect to spend more money on keyboards. Your preferences will differ from mine and you will want to try this and that.
- Be gentle on yourself when adopting to a new keyboard. I find that I usually tense up a lot more than when I type on a familiar keyboard. Ironically, this has the opposite effect and makes my wrists hurt more.

## On thumb clusters

Make sure that the thumb cluster is actually comfortable for you. I have a basically unused Corne because I realized too late that I don't like the thumb keys on it. For me the keys need to be further out like they are on the Kyria.

I find that I can make good use of 3 keys on each thumb, and this has become my preference. I use one as a key and two as layer toggles per hand. I can fit all the keys I could possibly need on the base layer and 4 additonal layers.

Lastly, I only noticed after many layout iteratons that you can assign different keys to the thumbs on each layer. It has been quite useful. Completely obvious once you see it to be honest!

## Other thoughts

- Having too many keys is not a real problem apart from asthetics. There is a big trend towards building 34-key keyboards. In retrospect, it is always easier to not use keys that are on your board instead of not having them. If I were to buy my first board, I would buy one with more than 34 keys.
- I do not remap any app specific keybindings to my new base layer. In `neovim` I try to use larger motions to navigate the text, and if I really need `hjkl`, I switch to my naviation layer.
- Developing your own keymap with QMK has become significantly easier since the introduction of the `qmk-userspace`. Make sure to periodically update your `qmk_firmware` folder to get the latest features.

I hope these tips were useful to you! Happy building!
