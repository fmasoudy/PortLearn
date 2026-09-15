# PortLearn Brand Package

## About the brand

The PortLearn brand pairs a rounded navy-and-teal monogram with the **PortLearn** wordmark. This package is the canonical home of that identity: vector and raster lockups, monochrome and dark-background variants, icon sizes, and a favicon — ready for documentation, GitHub, PyPI, and print contexts.

## Included assets
- `svg/` — crisp vector exports
- `png/primary/` — primary transparent PNG logos
- `png/monochrome/` — one-color versions
- `png/dark/` — white-on-dark versions
- `icons/` — app/icon sizes for GitHub, PyPI, and docs
- `favicon/portlearn-favicon.ico` — favicon
- `source/PortLearn_approved_concept.png` — design-concept reference

## Recommended usage
- Use `svg/portlearn-logo-horizontal.svg` for README headers and documentation.
- Use `svg/portlearn-logo-stacked.svg` for larger brand display.
- Use `icons/portlearn-icon-512.png` or `icons/portlearn-icon-256.png` for GitHub/PyPI.
- Use monochrome variants in academic documents when color is not suitable.

## Naming
The package uses **PortLearn** naming only.

## Usage guidance by variant

| Variant | Use for |
|---|---|
| Horizontal (`svg/portlearn-logo-horizontal.svg`, `png/primary/portlearn-logo-horizontal.png`) | README and documentation headers; any wide, short context where the mark sits left of the wordmark |
| Stacked (`svg/portlearn-logo-stacked.svg`, `png/primary/portlearn-logo-stacked.png`) | Large or hero display — covers, title slides, splash/landing sections |
| Stacked simple (`svg/portlearn-logo-stacked-simple.svg`, `png/primary/portlearn-logo-stacked-simple.png`) | Compact vertical contexts — narrow columns, sidebars, cards where the full stacked lockup is too tall |
| Wordmark (`svg/portlearn-wordmark.svg`, `png/primary/portlearn-wordmark.png`) | Text-led layouts — running headers/footers, colophons, list items where the mark is already established nearby |
| Tagline (`svg/portlearn-tagline.svg`, `png/primary/portlearn-tagline.png`) | Secondary lockup only — supporting text beneath an established horizontal or stacked lockup, never as a standalone logo |
| Icon (`svg/portlearn-icon.svg`, `icons/portlearn-icon-*.png`) | Avatars and square contexts — GitHub/organization avatar, package/docs favicon, app icons, small square placements |
| Monochrome (`png/monochrome/`) | One-color reproduction — academic documents, print, single-ink contexts (`-navy` on light grounds, `-white` on dark grounds) |
| Dark (`png/dark/`) | White-on-dark placement where an opaque navy field is wanted |

## Contrast guidance
- The primary horizontal lockup is navy/teal on a transparent background; it reads best on light grounds.
- On dark grounds use the dark variants (`png/dark/`) or the monochrome `-white` lockups (`png/monochrome/portlearn-logo-horizontal-white.png`).
- The root README header is adaptive on GitHub: it uses the primary horizontal logo with the monochrome-white horizontal as the dark-scheme fallback. On PyPI, the README text renders, but `readme-renderer` strips the `<source>` dark variant and PyPI does not host the relative repository image, so the header image degrades to alt text rather than a guaranteed logo.

## GitHub and PyPI presentation
- GitHub renders README images from relative repository paths; the root README uses `docs/assets/brand/png/primary/portlearn-logo-horizontal.png` (and the monochrome white variant for the dark scheme).
- **PyPI has no repository-file field that automatically sets a project avatar/logo.** The package description on PyPI is the long-description rendered from the README (via the `readme` metadata); the icon/avatar shown on PyPI is account-level, not set by any project metadata field. Do not fabricate such a field.
- The icon set (`icons/`) serves GitHub avatars, organization icons, documentation favicons, and app icons — it does not set a PyPI logo.
