# PoRacer — Google Play Console listing

> Drafted 2026-08-28 from the project itself, following the PoSumo conventions.
> Every field marked **TODO** needs a human in the console — either because it
> is a legal declaration, or because it needs an asset or URL that does not
> exist yet.

## 1. Create app

| Field | Value |
|---|---|
| App name | PoRacer: AI Creature Racing |
| Default language | English (United States) – en-US |
| App or game | Game |
| Free or paid | Free — **permanent**, a free app can never be switched to paid |
| Declarations | **TODO** — you must tick Developer Program Policies and US export laws yourself |

## 2. Application id

`com.punkoutersoftware.poracer`

Permanent once the first bundle is uploaded. Set it in the build **before** that upload.

## 3. Store listing

**App name** (27/30)

```
PoRacer: AI Creature Racing
```

**Short description** (74/80)

```
Watch creatures that taught themselves to run race each other. No scripts.
```

**Full description** (724/4000)

```
PoRacer is a race between physics creatures — worms, and rigged humanoids — that were never animated. Each one is a neural network trained with reinforcement learning until it worked out, on its own, how to move a body forward without falling over.

• Worms, humanoids and other creatures, each with a separately trained brain
• Every gait is learned from scratch — no keyframes, no motion capture
• Ragdoll physics: a bad step is a real stumble, not a canned animation
• Runs entirely on your device; nothing is uploaded anywhere

PoSumo, PoBox, PoCross, PoDance, PoFlag, PoFootball and PoSoccer are all built the same way: a physics simulation, and behaviour that was learned rather than written.

From Punkouter Software.
```

## 4. Graphic assets

| Asset | Requirement | Status |
|---|---|---|
| App icon | 512x512 PNG, under 1 MB | `StoreAssets/PlayStoreIcon_512.png` — ready |
| Feature graphic | 1024x500 PNG or JPEG | `StoreAssets/FeatureGraphic_1024x500.png` — ready |
| Phone screenshots | 2–8, portrait, each side 320–3840 px | **TODO** — capture from a device or emulator |
| Tablet screenshots | 7-inch and 10-inch | **TODO**, and only if you want tablet distribution |
| Promo video | YouTube URL | Optional — skip |

The launcher icon inside the app (`Assets/Icons/`) and this store icon are
deliberately different files: the launcher one is a squircle with transparent
corners because the OS masks it, the store one is full-bleed because Play
rounds it itself. Do not swap them.

## 5. Store settings

| Field | Value |
|---|---|
| App category | Game |
| Category | Racing |
| Tags | Racing, Simulation, Artificial Intelligence, Physics, Single player |
| Contact email | punkouter26@gmail.com |
| Contact website | **TODO** |
| Contact phone | Optional — leave blank |
| External marketing | No |

## 6. App content

| Section | Answer |
|---|---|
| Privacy policy | **TODO** — a public URL is required before you can release on any track |
| Ads | No, my app does not contain ads |
| App access | All functionality is available without any special access |
| Content ratings | See section 7 |
| Target audience | 13–15, 16–17, 18 and over |
| Appeals to children | No — keeps you out of the Families policy programme |
| News app | No |
| COVID-19 contact tracing | No |
| Government app | No |
| Financial features | None of these |
| Health apps | No |
| Data safety | See section 8 |

## 7. Content rating questionnaire (IARC)

| Question | Answer |
|---|---|
| Category | Game |
| Violence | None |
| Sexuality, nudity | None |
| Language | None |
| Controlled substances | None |
| Gambling, simulated gambling | None |
| Horror, fear | None |
| User interaction | None — no chat, no user-generated content, no social features |
| Shares user location | No |
| Shares personal information | No |
| Allows digital purchases | No |
| **Expected result** | Everyone / PEGI 3 |

The questionnaire is a legal declaration, so read each question yourself before
submitting — these are the answers the project supports, not a signature.

## 8. Data safety

| Question | Answer |
|---|---|
| Does your app collect or share any of the required user data types? | No data collected |
| Data shared with third parties | No data shared with third parties |
| Is data encrypted in transit? | Not applicable — nothing is transmitted |
| Can users request data deletion? | Not applicable — nothing is stored off-device |

> **Check this before you submit.** Unity bundles for these projects request
> `android.permission.INTERNET`, pulled in by the ML-Agents runtime rather than
> by anything the game does. The app makes no network calls at run time, so
> *no data collected* is the truthful answer — but Play does cross-check declared
> permissions against data-safety answers, so confirm the permission is either
> stripped from the release build or that you are comfortable explaining it.

## 9. Internal testing track

| Field | Value |
|---|---|
| Testers | **TODO** — an email list or a Google Group |
| Feedback channel | punkouter26@gmail.com |
| Release name | PoRacer: AI Creature Racing 1.0.0 (1) |
| Release notes | `First internal build.` |

The **first** bundle has to be uploaded by hand: Play will not accept an API
upload until the listing, content rating and data-safety forms are complete and
the track has testers. After that a service account can own every upload.

## 10. What this project still needs before it can produce that bundle

Nothing — the release pipeline is in place and a signed bundle has been built
and verified. See section 5 of this project's `CLAUDE.md`.
