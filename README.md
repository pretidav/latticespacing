# latticespacing

Personal blog, built with Jekyll and served by GitHub Pages. Math via KaTeX (`$inline$`, `$$display$$`).

## New post
Add `_posts/YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "My title"
category: physics
---
```

## Preview locally
```
bundle install
bundle exec jekyll serve
```
Then open http://localhost:4000/latticespacing/ (the site lives under `baseurl: /latticespacing`).
