# latticespacing

Personal blog.

## Images
Put each post's images in `assets/img/<post-file-name>/`, e.g. `assets/img/2026-11-17-hello-world/plot.png`. Then in the post:

```liquid
{% include figure.html src="/assets/img/2026-11-17-hello-world/plot.png" caption="Optional caption" %}
```
or plain Markdown (the `relative_url` filter keeps the `/latticespacing` prefix working):
```
![alt text]({{ '/assets/img/2026-11-17-hello-world/plot.png' | relative_url }})
```
