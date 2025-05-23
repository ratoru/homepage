---
title: "Finding the perfect ergonomic keyboard for you"
description: "Everything you need to know to pick the right ergonomic keyboard."
tags: ["coding", "keyboard"]
publishDate: 2023-07-22
featured: true
---

![keyboards](../../assets/blog/keyboard/keyboard.webp)

I hated keyboards until I discovered ergonomic keyboards.

My future was filled with countless hours of typing on poorly designed, regular keyboards. The light at the end of the tunnel: wristpain.
Ergonomic keyboards, on the other hand, are designed to prevent pain.
Unfortunately, the ergonomic keyboard rabbit hole is a daunting world for newcomers.

This guide will walk you through everything you need to know about ergonomic keyboards.

## The basic idea

The goal of ergonomic keyboards is to make typing strainless and prevent repetitive strain injury (RSI).

- Split the keyboard for better [shoulder and wrist posture](https://www.youtube.com/watch?v=1C2bJkzIaPE&t=3s).
- Introduce staggered key columns so that fingers move as their natural motion dictates.
- Add two thumb clusters so that each thumb presses multiple keys.
- Reprogram your keys to create a custom keyboard layout and special keys.
- Optionally: Reduce the number of keys, so that you barely move your fingers while typing.
- Optionally: Tilt the each half of the keyboard to avoid pinching the nerve in your lower arm.

If these points sound appealing to you, you are in the right place.

## DIY Keyboards

Start by getting a general overview of [keyboard terminology](https://blog.keeb.io/first-timers/).

There are many open source, DIY keyboard designs that follow these rules. You can use [this website](https://jhelvy.shinyapps.io/splitkbcompare/) to print out the layout of each keyboard to check whether you would enjoy them. I highly recommend putting your fingers on the paper to figure out whether pressing the keys is comfortable.

You will need a range of [components](https://docs.splitkb.com/hc/en-us/articles/6269948925084-Aurora-Build-Guide-2a-What-s-what-) and soldering equipment. I would expect the total cost to land somewhere between $150 and $300 dollars. The biggest price differentiator is probably which keycaps you buy. I went for a relatively basic setup and payed $200. This is much cheaper than prebuilt ergonomic keyboards like the [Moonlander](https://www.zsa.io/moonlander/).

[Awesome Split Keyboards](https://github.com/diimdeep/awesome-split-keyboards) on GitHub has a comprehensive list of keyboards and collection of useful links.

### Things to be aware of

- Use sockets that let you replace your switches.
- Socket your microcontroller. These can easily break and you don't want to have to replace the PCB board.
- How aggressive is the pinky column stacked? Depending on your hand some of the layouts might be uncomfortable for the pinky finger.
- There is a learning curve learning this new keyboard layout.

### Keyboards that stand out to me

The keyboard in the picture is the [Kyria](https://github.com/splitkb/kyria). It's sold on [splitkb.com](https://splitkb.com/products/kyria-rev3-pcb-kit?variant=43642869580035). The Kyria has a very aggressive pinky stagger, which I like. Additionally, `splitkb` has a buy guide that walks you through all the components you have to buy, which helps a novice builder a lot.

Other great keyboards are the ...

- [Iris](https://keeb.io/products/iris-keyboard-split-ergonomic-keyboard)
- [Corne](https://github.com/foostan/crkbd/tree/main)
- [Lily58](https://github.com/kata0510/Lily58)

## Prebuilt keyboards

Of course, you don't have to build your own ergonomic keyboard. Prebuilt keyboards can be great for people that don't want to to bother assembling a keyboard or compiling their own firmware. Note that they are more expensive, though.

- Expensive pre-built option: [Moonlander](https://www.zsa.io/moonlander/buy)

## Switches

You can usually buy either **MX switches** or **CHOC switches**.

MX switches are the big keys you see on normal mechanical keyboards. There are many colorful caps to customize your keyboard with.

CHOC switches are very low profile, thin switches similar to laptop keyboards. There are not that many options for personalizing the caps.

Beware of the sound when buying. A lot of people in the community enjoy having loud switches, but I want mine to be no louder than the MacBook's keys. Otherwise, your typing will be audible over phone calls, your roommates will be annoyed, and you can't use your keyboard at work. The [Durock Dolphins](https://keeb.io/products/durock-dolphin-silent-linear-switches) seem like a good MX-compatible option. I bought the [Gazzew Bobagum Silent Linear Switch](https://splitkb.com/products/gazzew-bobagum-silent-linear-switch?_pos=8&_sid=4908b0746&_ss=r), which works great so far.

### Switch Types

| Type    | Feeling            | Noise    |
| ------- | ------------------ | -------- |
| Linear  | Smooth, Consistent | Quiet    |
| Tactile | Bumpy              | Moderate |
| Clicky  | Bumpy              | Loud     |

### Rotary Encoders

Some of the custom keyboards support rotary encoders. They are small knobs that you can turn or press to scroll, regulate volume, etc.. Read [this guide](https://docs.splitkb.com/hc/en-us/articles/360010513760-How-can-I-use-a-rotary-encoder-) to see what they are useful for and whether you would like to buy one.

## Keycaps

Keycaps allow for a lot of customization. They come with different profiles, colors, and manufacturing quality. Obviously, you want to buy keycaps that fit your switch type. Here is a good video explaining all the [different MX profiles](https://www.youtube.com/watch?v=14bQeqhlTNM) you can buy.

In general, blank keycaps are easy to find. Keycaps with letters and the special symbols are harder to find. If you find a good vendor, you can also get a cool color scheme. [Drop.com](https://drop.com/home) has beautiful, but expensive, keycaps. For CHOC switches I can recommend the [work louder keycaps](https://worklouder.cc/keycaps/). You can also get some cheap ones for either switch type off of Amazon.

Since custom keycaps are a niche market, people sometimes organize group orders to make them affordable. These can take a year to ship, so pay attention to what you buy. There can be very long wait times.

### Material

- **ABS**: Great for backlit keyboards. Very customizable because they are easy to produce. More shock resistant. Will develop a shine after long use. I think this is what the Macbook keyboard uses. (Might be more silent?)
- **PBT:** More involved production. Your fingers rest on the plastic instead of the paint, so they won't develop a shine. Harder to find customizations for. Doesn't look as good with RBG backlit boards.

**Double shot** key caps are made using two separate pieces of plastic which get moulded together. This makes the keys and imprint last longer.

### Profiles (= shapes)

:::note
Nothing beats testing switches and profiles in a store.
:::

I think it's best to look at a chart of profiles online. Note that uniform profiles make it easier to assemble a custom layout because you don't have to worry about which direction the keycaps shape when putting it onto your board. CHOC keycaps are all uniform, so you won't have any trouble there.

I read good reviews about the MRT shaped keys, but if going for a Colemak-DH layout, then you should probably look into DSA.

## Assembly

![keyboard-no-keys](../../assets/blog/keyboard/keyboard-no-keys.webp)

Follow the assembly guide of your keyboard. They are usually very good. I had very little previous soldering experience and managed. Assembly will take time, though. I highly recommend asking questions in the corresponding community. I asked questions in the `splitkb` Discord server and received a lot of help.

## Accessories

If you are new to mechanical keyboards, you will notice that you cannot rest your wrist on your laptop anymore. Therefore, you might want to get comfortable floating your wrists. Otherwise, you should look into wrist rests.

Do you want to completely max out the ergonomics stat of your keyboard? Then, you should consider tenting your keyboard. Depending on the board you built you have different options. Your board's case might have come with tenting functionality, e.g. using a tri-pod or screws. Otherwise, search GitHub and [Thingiverse](https://www.thingiverse.com/) for tentable cases to 3D print.

## Keyboard Layouts

You will need one. Be warned, though. A custom keyboard layout is another rabbit hole. While you wait for your order to deliver, I recommend you get started on designing one. Afterwards, all that's left to do is flash your firmware. Here you can see the keymap for my Aurora Sweep:

![sweep-hrm-layout](../../assets/blog/layout/sweep-hrm.webp)

I'll show you how to make your own in my [3-part guide on custom keyboard layouts](https://ratoru.com/blog/choose-the-right-base-layout).
