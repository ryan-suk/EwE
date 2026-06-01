# GitHub and Zenodo DOI Steps

This folder is ready to upload to the existing GitHub repository `ryan-suk/EwE`, preferably under the subfolder `reddit-hpv-online-forum-analysis/`.

## Before Upload

1. Update `CITATION.cff` with the final creator names and ORCID identifiers if available.
2. Update `.zenodo.json` with the final creator names and ORCID identifiers if available.
3. Confirm that no row-level Reddit-derived files are included. This package intentionally excludes raw Reddit extracts, `seed_frame.csv`, and `sentence_predictions_v2.csv`.

## GitHub Upload

Upload the contents of this folder to:

```text
https://github.com/ryan-suk/EwE/tree/main/reddit-hpv-online-forum-analysis
```

The public GitHub version can use the included SVG figures for direct rendering. PNG versions are also available in the local release archive.

## DOI Minting Through Zenodo

DOI minting has been completed for the Zenodo-enabled GitHub release `v1.0.2`.

- Version DOI: `10.5281/zenodo.20482126`
- Concept DOI for all versions: `10.5281/zenodo.20481784`

1. Sign in to Zenodo with the GitHub account that owns the repository.
2. Go to Zenodo's GitHub integration page and enable archiving for the repository.
3. In GitHub, create a release, for example `v1.0.0`.
4. Zenodo will archive the release and mint a DOI.
5. Copy the DOI back into the GitHub repository README and citation metadata if desired.

## Suggested Release Title

```text
Reddit HPV Online Forum Preliminary Analysis v1.0.0
```

## Suggested Release Notes

```text
Initial public aggregate release for the preliminary Reddit HPV online forum analysis.

This release includes aggregate prevalence tables, figures, documentation, run metadata, and reproducible pipeline code. Raw Reddit posts/comments and row-level sentence prediction files are excluded to reduce privacy risk for sensitive health narratives.
```
