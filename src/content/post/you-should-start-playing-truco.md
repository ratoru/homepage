---
title: "You should start playing Truco"
subtitle: "An Introduction"
description: "The Argentinean card game combines the trick based gameplay of Skat and Schafkopf with the betting complexity of Poker."
tags: ["truco"]
publishDate: 2026-02-16
---

Truco is my favorite card game.

Its mix of trick based gameplay and Poker style betting creates a beautiful
balance of luck and skill expression. One day you'll casually play with friends
over a few beers, another you'll debate whether you've been properly realizing your hand
equity. Learning Truco can be daunting, so I curated and explained all the rules for you here.

All you need is a deck of Spanish playing cards to get started with this beloved Argentinean card game.

## Some background

You will love Truco if any of the following is true...

- You enjoy playing card games like Schafkopf, Skat, or Bridge.
- You enjoy the betting in Poker, but want a simpler version that can be played without money.
- You are looking for a fun game to play with only a deck of cards.
- You like games where game-theory optimal play is not obvious.

Truco is a card game that's very popular in many parts of South America, with
each region often having their own rules. This guide focuses on the Argentinean
version since that's where I learned the game and first started playing it
seriously.

While I'm generally a stickler for rules, I found online rulesets to be
incomplete or not in service of a competitive game[^1]. Therefore, I curated
the rules below based on two principles.

[^1]:
    Card games are ultimately based on luck. Nonetheless, a better player
    should be able to beat a worse one in expectation. Therefore, I decided to
    adjust rules that offset the balance between luck and skill to make the game
    more fun _for me_.

1. The ruleset should be complete and easy to follow.
2. The ruleset should make the game fun.

So, read through this guide, start playing with your friends, and adopt the game
to your needs. I am by no means a Truco-purist.

## The rules

:::note[Official Rules]
Truco has **a lot** of rules, house rules, and variations. The closest ruleset
to how I learned the game in Argentina can be found on
[pagat.com](https://www.pagat.com/put/truco_ar.html). If you'd like to stick
with the official rules, you can always reference that page.
:::

| Basic Terminology | Definition                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| Trick             | One round of each player placing down 1 card each.                                                      |
| Suit              | The color of a given card.                                                                              |
| Trump Card        | A special card that has a higher value than other cards.                                                |
| Round             | All the tricks and betting that happens between each shuffle.                                           |
| Game              | All the rounds that are played until a team wins.                                                       |
| `Reserved Words`  | Words that signal a specific action in the game. Saying them out loud commits your team to that action. |

You need a deck of **Spanish playing cards** to play. The deck will have 40 cards: no 8s, 9s or
13s. Jack and Queen become 11 and 12.

Truco is played with **2, 4, or 6 players**[^rec]. The teams are predetermined and do not change during a round.

[^rec]: I highly recommend 4 or 6 players!

- A 2 person game is played 1v1.
- A 4 person game is played 2v2. Team membership alternates around the table. Therefore, on a round table you sit across from your partner.
- A 6 person game is played 3v3. Team membership alternates around the table. See [6 Players](#6-players) for more info.

### The basic structure of the game

1. The first team to 30 points wins.
2. Betting can occur throughout the round to increase the point value of the round.
3. After each round, the points are added to the teams' totals, the dealer rotates, and new cards are dealt.
4. A round (ignoring betting) consists of...
   1. Each player gets dealt 3 cards.
   2. 3 tricks are played one after another.
   3. The person to the right of the dealer (`el mano`) starts.
   4. Counter-clockwise and one after another, each player plays one card face up[^3]. You can play whatever card you want.
   5. The winner starts the next trick. If it was a tie, the same person starts.

[^3]: Traditionally, cards are placed face up directly in front of you. This lets everyone see all previously played cards. I prefer placing all cards on top of each other in the middle, and giving the winner the trick face down. This rewards card counting.

### Card order

```text
1 of Swords
1 of Clubs
7 of Swords
7 of Coins
3 - 1
12 - 4
```

Truco has 4 trump cards: 1 of Swords, 1 of Clubs, 7 of Swords, and 7 of Coins. These are the 4 highest cards in the game. After that, you have 3, 2, the remaining 1s, 12, 11, 10, 7, ..., 4.

Aside from the trump cards, suits do not matter. A trick that consists of two cards of the same value will tie.

### The deal

- The dealer shuffles and lets the person to the left cut the deck. The person to the left can tap the deck to skip the cut.
- Cards are dealt one-by-one counter-clockwise until each player has 3 cards.
- The deck is placed to the right of the dealer[^deal]. The player to the right of the dealer is called `el mano`. `El mano` opens the first trick.
- `El mano` will become the dealer for the next round in 2 and 4 player games. In 6 player games, the same dealer will deal twice.

[^deal]: Do not forget this! It's very useful to resolve ties and keep the game moving quickly.

### Truco Betting

Each round of Truco has two separate betting "games". The first, `Truco` betting, bets on
the actual trick playing of the game. In other words, the winner of the tricks
will win the `Truco` bet.

All the betting terminology uses reserved words. Saying any of the words marked
as `reserved` will commit you to that action. Be careful, someone might trick
you into saying one of them.

By default a round of tricks is worth **1 point**. A team can raise the value of the round as follows.

- If nothing has been called, `Truco` will raise the stakes to **2 points**. The other team can accept with `Quiero`, fold using `No Quiero`, or re-raise using `Retruco`. Folding will give the other team 1 point.
- If `Truco` has been called previously, you can re-raise to **3 points** by calling `Retruco`. Once again, the other team can accept with `Quiero`, fold using `No Quiero`, or re-raise using `Vale Quatro`. Folding will give the other team 2 points.
- If `Retruco` has been called previously, you can re-raise to **4 points** by calling `Vale Quatro`. The other team can only accept or fold. No further raising is possible. Folding will give the other team 3 points.

| Betting Sequence              | Points if Accepted | Points if Folded |
| ----------------------------- | ------------------ | ---------------- |
| Truco                         | 2                  | 1                |
| Truco + Retruco               | 3                  | 2                |
| Truco + Retruco + Vale Quatro | 4                  | 3                |

`Truco` can be called **at any point**! This means you can call Truco before you play a card, after you play a card, or even during someone else's turn.

Only the team that accepted a raise can re-raise. They can re-raise at any point. Therefore, team A can accept B's `Truco` with `Quiero` and one trick later A can call `Retruco`, but B cannot.

If a team folds by saying `No Quiero`, the round stops immediately. No team needs to show any of their remaining cards - unless you won `Envido`.

### Envido Betting

The other form of betting is called `Envido`. This is a separate "game" where you
bet on the card values in your hand. A round of play always has the trick based
`Truco` game, but if no `Envido` is called, there will be no `Envido` points.
If an `Envido` bet is accepted, the player with the highest `Envido` score wins
the bet.

You calculate your score as follows:

- Each card is worth its number, except for 10, 11, and 12 which are worth 0.
- Pick two cards in your hand and add them together. That will be your score.
- If you have two cards of the same suit, you can add 20 to your score.

Thus, the highest possible Envido is a suited 6 and 7, which will score as $6 + 7 + 20 = 33$.
A suited 11 and 5 will score as $0 + 5 + 20 = 25$[^score].

[^score]: As a rule of thumb, I would consider anything 28 and higher a good `Envido`.

You can bet `Envido` only before you play your first card in the first trick.
You can have your partner call `Envido` if you already played your card. Letting
others play cards before you call Envido can give you additional information
about their hand.

:::caution[Envido offsets the balance of the game]
Envido re-raising is a little complicated, and I am still figuring out the
ruleset I would like to play with. For now, I recommend playing with just
`Envido` and re-raise with `Real Envido` (similar to `Retruco`) for 3 points
instead of using the table below.
:::

The exact rules of betting are as follows. Please remember that these are `reserved` words.

- If you have not played any cards this round and no Envido bet has been called, you can call `Envido` to play for **2 points**. The opposing team can accept with `Quiero`, fold with `No Quiero`, and re-raise. Folding will give the other team 1 point.
- If you have not played any cards this round and no Envido bet has been called, you can call `Real Envido` to play for **3 points**. The opposing team can accept with `Quiero`, fold with `No Quiero`, and re-raise. Folding will give the other team 2 points.
- If you have not played any cards this round and no Envido bet has been called, you can call `Falta Envido` to play for **as many points as the leading team needs to win**[^6]. The opposing team can accept with `Quiero`, fold with `No Quiero`. Re-raising is not possible. This is akin to going all-in. Folding will give the other team 1 points.

[^6]: I dislike `Falta Envido` and being able to raise Envido for more points than Truco because, ultimately, I want to play a trick based game - not a hand drawing game with tricks on the side. The game is called `Truco` after all and not `Envido`.

Traditionally, you can re-raise as long as the re-raise is at least as large as
the previous bet. I listed **all possible sequences of bets** in the table below.
Note, that `Falta Envido` could always be called as a re-raise action. Falta
Envido overrides any previous Envido raising, i.e. if the leading team needs 2
points, but `Real Envido` was previously called, the Envido bet is still only
worth 2 points.

| Betting Sequence              | Points if Accepted | Points if Folded | Explanation          |
| ----------------------------- | ------------------ | ---------------- | -------------------- |
| Envido                        | 2                  | 1                |                      |
| Real Envido                   | 3                  | 2                |                      |
| Envido + Envido               | 4                  | 3                | 2 + 2, 2 + 1         |
| Envido + Real Envido          | 5                  | 4                | 2 + 3, 2 + 2         |
| Real Envido + Real Envido     | 6                  | 5                | 3 + 3, 3 + 2         |
| Envido + Envido + Real Envido | 7                  | 6                | 2 + 2 + 3, 2 + 2 + 2 |

If an `Envido` bet gets accepted, the last person to raise starts by calling out
their number. Then, counter-clockwise players can either say their number or
`son buenas`. `Son buenas` admits that you cannot beat the current highest
Envido. Once everyone has responded, the winner is clear and the trick playing
continues. Ties go to the player closest to the `mano`.

At the end of the round, you may be asked to show your Envido cards. If you said
the wrong number earlier, the other team automatically wins the Envido bet.

`Envido` is resolved separately from `Truco`. This means that if a team rejects
the `Envido` bet with `No Quiero`, the trick based `Truco` game continues as
normal.

`Envido` takes precedence over `Truco`. This means that you can respond to a
`Truco` bet from your opponents by calling `Envido`. In that case, the other
team will have to respond to your `Envido` bet first, before you have to respond
to their `Truco` bet. Obviously, you can only do this if someone on your team
hasn't played a card at all yet.

As you can see, `Envido` gives weaker hands a chance to earn points through
clever betting. This is one of the reason why Truco is a very dynamic game where
good strategy can shine.

### Resolving ties

After a tie, the same player that opened the previous trick starts again.

- If the first trick ties, the second trick is winner take all.
- If the second trick ties or trick 1 and 2 tie, the third trick is winner take all[^5].
- If the third trick ties or all 3 tricks are ties, the `mano` wins.
- If two `Envido` scores are the same, the player - not team - closer to the `mano` wins.

[^5]: Traditionally, the winner of the first trick would win here. I like the tension of the 3rd trick.

### Scoring

Points for `Envido` and `Truco` are added to the team's total after the round
finishes. This means that if a game is tied at 29-29, team _A_ calls `Envido`
and team _B_ calls `Truco`, and both are rejected, the score will be
`30-30`[^4].

[^4]: If you decide to score Envido as it is called, team _A_ would win here.

The first 15 points are called `malas` and the next 15 points are called
`buenas`. So, a score of `3 malas` to `5 buenas` corresponds to a score of 3-20.
This means you can keep track of the score using 15 beans per team, resetting
the bean count once you cross from `malas` to `buenas`.

### Resigning

At any point a player can decide to resign by placing their cards face down on the deck. At this point they can no longer win any tricks. Their partner could continue playing alone.

If you called `Envido`, you can be asked to prove your `Envido` score even if you throw away your cards. In that case, you must show the `Envido` cards.

### Flor betting

I **do not** play with Flor. It's a complicated form of betting when you have 3
cards of the same suit. Unlike Envido, Flor must be called when you have it and
must not be called otherwise. This makes it a purely luck based mechanic, which
I decided to remove from the game. Playing without Flor is a variant called `sin
Flor` and quite common.

## 6 Players

A game of 6 players is played by alternating rounds of 3v3 and three 1v1s. Both
rounds are played with the standard Truco rules.

First, a round of 3v3 is played as usual. Then, three 1v1s are played. These
1v1s are played against the player across from you (3 to your right). The dealer
deals cards for all players. Then, each 1v1 is played one after another. Each
1v1 will have its own `Truco` and `Envido` betting. This means that later 1v1s
will know what cards have already been played. Each player represents their team
and the scores get added to the teams score after all hands have been played.

To ensure that the starting positions are fair, the dealer needs to deal both
the 1v1 and the 3v3 before the dealer rotates. Otherwise, the same team would
always be in position for the 3v3.

## A quick recap

- Games are played to 30. You play with the person across from you.
- Each round consists of 3 tricks played Bo3.
- `Mano` starts the first trick. The winner of a trick starts the next one.
- Each round has betting for `Truco` and `Envido`.
- After a bet you can be fold (`No Quiero`), accept (`Quiero`), or (usually) reraise.

## Signals

This is the standardized set of signals that you can use to communicate your
hand to your partner. You should learn these and use them as much as you can.
They are an essential part of the game. If both teams use the same signals, you
can catch the other team's signals to give you an advantage.

| Card                      | Signal                          |
| ------------------------- | ------------------------------- |
| 1 of Swords               | Raise an eye-brow               |
| 1 of Clubs                | Wink (either eye)               |
| 7 of Swords or 7 of Coins | Move lips to a side of the face |
| 3                         | Bite your lip                   |
| 2                         | Blow a kiss                     |
| 1                         | Open your mouth                 |
| Bad cards                 | Close both eyes                 |
| Good `Envido`             | Tilt your head                  |

If both teams agree to it, each team can instead create their own set of
signals. Now, the game becomes a little bit more about code cracking than
secrecy.[^2]

[^2]:
    You might run into someone that despises custom signals. But, this is one of
    these instances where I think the rules should allow for whatever is fun for
    you.

## An example round

```text
TODO: mermaid chart
|-- B --|
|       |
Y       X
(deck)  |
|-- A --|
```

- _Y_ is the dealer. _A_ is `mano`.
- Signals are exchanged. Specifically, _X_ signals to _Y_ to call `Envido` by tilting the head.
- _A_ plays the 6 of Coins.
- _X_ plays the 7 of Cups.
- _B_ plays the 3 of Cups and calls `Truco`.
- _Y_ responds by calling `Envido`. (_Y_ has not played a card yet, and `Envido` takes precedence.)
- _A_ declines by responding `No Quiero`. (1 point to team _XY_)
- Now, _XY_ needs to respond to the `Truco` bet. They accept with `Quiero` and _Y_ plays the 3 of Swords.
- The trick ties, so _A_ will start again. _A_ plays the 2 of Clubs.
- _X_ re-raises by calling `Retruco`. _B_ declines with `No Quiero`. (2 points for _XY_)
- _XY_ gets 3 points total, and _A_ becomes the dealer.

## Just start playing

You now have all the necessary knowledge to start playing. If you want to learn
some strategy, I suggest you read my introduction to Truco strategy.
