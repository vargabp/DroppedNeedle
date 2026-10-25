"""Provider-independent grouping and stable local-album continuity."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from models.identification import (
    ExistingAlbumMembership,
    GroupingTrack,
    ProposedLocalAlbum,
)

# M-03 (Phase 2 step 2.1) disc-folder spellings: cd/disc/disk/vol/volume
# with optional parentheses around the word and/or the number (CD1,
# Disc-02, Disk 1, Disc(1), (CD1), Volume 2, Vol.1). The anchors require a
# whole-segment match, so suffixed segments (CD01 bonus, Volume 1
# Remastered) stay per-folder. The disc_number TAG stays ignored here
# (directory evidence only - out of scope, recorded for
# .dev-notes/Plans/LibraryFindings-All/06-risks.md).
_DISC_DIRECTORY = re.compile(
    r"^\(?(?:cd|disc|disk|vol|volume)\)?[\s._-]*\(?0*(\d+)\)?$", re.IGNORECASE
)


def _match_disc_directory(segment: str) -> re.Match[str] | None:
    """Disc-folder match with balanced parens: the shared regex is
    permissive about optional parens, so unbalanced spellings (``(CD1``,
    ``CD1)``) are rejected here at the grouping call sites."""
    match = _DISC_DIRECTORY.match(segment)
    if match is None:
        return None
    depth = 0
    for char in segment:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
    if depth != 0:
        return None
    return match
_SPACE = re.compile(r"\s+")
# M-05 (Phase 1 step 1.8): a merge anchored only on parsed evidence is
# provisional, low-confidence - filename evidence must lose ties to tags.
# Groups holding any tag-provenance album claim keep the tagged reason.
PROVISIONAL_PARSED_GROUP = "PROVISIONAL_PARSED_GROUP"
# Single source shared with the staged grouping path: the staged service
# imports this name, so the small-path and staged-path caps stay identical.
CONTINUITY_COMPONENT_EDGE_LIMIT = 512


def normalize_group_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    return _SPACE.sub(" ", normalized).casefold()


def _display_consensus(values: list[str], fallback: str) -> str:
    usable = [value.strip() for value in values if value.strip()]
    if not usable:
        return fallback
    folded_counts = Counter(normalize_group_value(value) for value in usable)
    selected = min(
        folded_counts,
        key=lambda value: (-folded_counts[value], value),
    )
    return min(value for value in usable if normalize_group_value(value) == selected)


def _directory_context(track: GroupingTrack) -> tuple[str, str | None]:
    parent = PurePosixPath(track.relative_path).parent
    match = _match_disc_directory(parent.name)
    if match:
        if str(parent.parent) not in ("", "."):
            return str(parent.parent), match.group(1)
        # M-03 top-level rule: a root-level disc folder (CD1/x.flac, no
        # grandparent) folds to the "." root group instead of standing alone
        # on its directory name. It groups with root siblings sharing album
        # tags via _fold_top_level_disc_groups below; a lone top-level disc
        # folder keeps a tag-keyed "." group (never dropped).
        return ".", match.group(1)
    return (str(parent) if str(parent) else "."), None


def grouping_directory(relative_path: str) -> str:
    parent = PurePosixPath(relative_path).parent
    if _match_disc_directory(parent.name):
        if str(parent.parent) not in ("", "."):
            return str(parent.parent)
        # M-03 top-level rule (mirror of _directory_context): root-level
        # disc folders belong to the "." root grouping context. Centralized
        # here so all grouping_directory callers inherit the same fold.
        return "."
    return str(parent) if str(parent) else "."


def _fold_top_level_disc_groups(
    groups: dict[tuple[str, str, str], list[GroupingTrack]],
    reasons: dict[tuple[str, str, str], str],
) -> None:
    """Merge top-level disc groups into same-tag sibling groups (M-03).

    The "." fold above unites top-level disc tracks that share one root
    context, but a top-level disc folder may also sit beside a sibling
    folder carrying the same album tags (CD1/01.flac next to Album/02.flac).
    Those key under different directories, so without this pass the disc
    folder would stand alone on its directory name. A "."-keyed
    COMPATIBLE_DISC_DIRECTORIES group therefore joins the first-sorted
    tagged group with the same album fold and artist partition under a
    different directory. Genuine root files (CONSISTENT_ALBUM_TAGS) never
    move, untagged (fold "") groups never move, and the disc fold never
    overrides differing album tags (the multi-disc negative stays split).
    """
    sources = sorted(
        key
        for key in groups
        if key[0] == "."
        and key[1]
        and reasons.get(key) == "COMPATIBLE_DISC_DIRECTORIES"
    )
    for source in sources:
        if source not in groups:
            continue
        hosts = sorted(
            key
            for key in groups
            if key != source
            and key[0] != "manual"
            and key[1] == source[1]
            and key[2] == source[2]
            and key[1]
        )
        if not hosts:
            continue
        target = hosts[0]
        groups[target].extend(groups.pop(source))
        del reasons[source]


def _is_parsed_anchored(members: list[GroupingTrack]) -> bool:
    """True when the album identity rests solely on parsed evidence.

    M-05 (Phase 1 step 1.8): at least one parsed-provenance album claim and
    no tag-provenance album claim - the merge is provisional,
    low-confidence. Absent/placeholder members abstain rather than anchor:
    they never rescue a parsed anchor back to a tagged reason (only a
    genuine tag claim does), and pre-vintage rows without any parsed anchor
    keep their tagged reason untouched.
    """
    return any(
        member.album_title_provenance == "parsed" for member in members
    ) and all(member.album_title_provenance != "tag" for member in members)


def _hungarian_min(cost: list[list[int]]) -> list[int]:
    size = len(cost)
    if size == 0:
        return []
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    infinity = 10**18
    for row in range(1, size + 1):
        p[0] = row
        column = 0
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            source = p[column]
            delta = infinity
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                candidate = (
                    cost[source - 1][candidate_column - 1]
                    - u[source]
                    - v[candidate_column]
                )
                if candidate < minimum[candidate_column]:
                    minimum[candidate_column] = candidate
                    way[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    assignment = [0] * size
    for column in range(1, size + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def _sparse_single_assignments(
    rows: list[int],
    columns: list[int],
    overlaps: Counter[tuple[int, int]],
    ordered_existing: list[ExistingAlbumMembership],
    ordered_proposed: list[ProposedLocalAlbum],
) -> dict[str, tuple[str, str]]:
    """Greedily match one best unmatched edge at a time.

    F-14 small-path mirror of the staged sparse fallback
    (``apply_next_sparse_grouping_continuity``): repeated best-single-edge
    picks in overlap-DESC, existing-order, proposed-order sequence instead
    of one dense Hungarian matrix. Zero-overlap pairs never match, and the
    tie rule counts equal-overlap edges among still-unmatched neighbours.
    """
    matched_rows: set[int] = set()
    matched_columns: set[int] = set()
    retained: dict[str, tuple[str, str]] = {}
    while True:
        pick: tuple[int, int, int] | None = None
        for row in rows:
            if row in matched_rows:
                continue
            for column in columns:
                if column in matched_columns:
                    continue
                value = overlaps[(row, column)]
                if value == 0:
                    continue
                if pick is None or value > pick[0]:
                    pick = (value, row, column)
        if pick is None:
            return retained
        _, row, column = pick
        old_best = max(
            overlaps[(row, candidate)]
            for candidate in columns
            if candidate not in matched_columns
        )
        new_best = max(
            overlaps[(candidate, column)]
            for candidate in rows
            if candidate not in matched_rows
        )
        tied = (
            sum(
                overlaps[(row, candidate)] == old_best
                for candidate in columns
                if candidate not in matched_columns
            )
            > 1
            or sum(
                overlaps[(candidate, column)] == new_best
                for candidate in rows
                if candidate not in matched_rows
            )
            > 1
        )
        matched_rows.add(row)
        matched_columns.add(column)
        old = ordered_existing[row]
        new = ordered_proposed[column]
        retained[new.grouping_key] = (
            old.local_album_id,
            "CONTINUITY_TIE_BROKEN" if tied else "MAXIMUM_TRACK_OVERLAP",
        )


def assign_album_continuity(
    existing: list[ExistingAlbumMembership],
    proposed: list[ProposedLocalAlbum],
) -> list[ProposedLocalAlbum]:
    """Choose the maximum-total-overlap one-to-one continuity assignment."""
    ordered_existing = sorted(
        existing, key=lambda album: (album.created_at, album.local_album_id)
    )
    ordered_proposed = sorted(proposed, key=lambda album: album.grouping_key)
    if not ordered_proposed:
        return []

    track_to_existing: dict[str, list[int]] = defaultdict(list)
    for row, album in enumerate(ordered_existing):
        for track_id in album.track_ids:
            track_to_existing[track_id].append(row)
    overlaps: Counter[tuple[int, int]] = Counter()
    existing_edges: dict[int, set[int]] = defaultdict(set)
    proposed_edges: dict[int, set[int]] = defaultdict(set)
    for column, album in enumerate(ordered_proposed):
        for track_id in album.track_ids:
            for row in track_to_existing.get(track_id, []):
                overlaps[(row, column)] += 1
                existing_edges[row].add(column)
                proposed_edges[column].add(row)

    retained: dict[str, tuple[str, str]] = {}
    visited_existing: set[int] = set()
    for initial in sorted(existing_edges):
        if initial in visited_existing:
            continue
        component_existing: set[int] = set()
        component_proposed: set[int] = set()
        pending_existing = [initial]
        while pending_existing:
            row = pending_existing.pop()
            if row in component_existing:
                continue
            component_existing.add(row)
            for column in existing_edges[row]:
                if column in component_proposed:
                    continue
                component_proposed.add(column)
                pending_existing.extend(proposed_edges[column] - component_existing)
        visited_existing.update(component_existing)
        rows = sorted(component_existing)
        columns = sorted(component_proposed)
        if sum(len(existing_edges[row]) for row in rows) > (
            CONTINUITY_COMPONENT_EDGE_LIMIT
        ):
            # F-14: components past the cap fall back to sparse single
            # assignment instead of a dense O(n^3) matrix, mirroring the
            # staged `_load_continuity_component` None + sparse path.
            for fallback_key, match in _sparse_single_assignments(
                rows, columns, overlaps, ordered_existing, ordered_proposed
            ).items():
                retained[fallback_key] = match
            continue
        size = max(len(rows), len(columns))
        maximum = max(
            overlaps[(row, column)] for row in rows for column in existing_edges[row]
        )
        weights = [
            [
                overlaps[(rows[row], columns[column])]
                if row < len(rows) and column < len(columns)
                else 0
                for column in range(size)
            ]
            for row in range(size)
        ]
        assignment = _hungarian_min(
            [
                [maximum - weights[row][column] for column in range(size)]
                for row in range(size)
            ]
        )
        for component_row, component_column in enumerate(assignment[: len(rows)]):
            if component_column >= len(columns):
                continue
            row = rows[component_row]
            column = columns[component_column]
            value = overlaps[(row, column)]
            if value == 0:
                continue
            old = ordered_existing[row]
            new = ordered_proposed[column]
            old_best = max(
                overlaps[(row, candidate)] for candidate in existing_edges[row]
            )
            new_best = max(
                overlaps[(candidate, column)] for candidate in proposed_edges[column]
            )
            tied = (
                sum(
                    overlaps[(row, candidate)] == old_best
                    for candidate in existing_edges[row]
                )
                > 1
                or sum(
                    overlaps[(candidate, column)] == new_best
                    for candidate in proposed_edges[column]
                )
                > 1
            )
            retained[new.grouping_key] = (
                old.local_album_id,
                "CONTINUITY_TIE_BROKEN" if tied else "MAXIMUM_TRACK_OVERLAP",
            )
    return [
        ProposedLocalAlbum(
            grouping_key=album.grouping_key,
            title=album.title,
            album_artist_name=album.album_artist_name,
            track_ids=album.track_ids,
            reason_code=album.reason_code,
            retained_album_id=(retained.get(album.grouping_key) or (None, None))[0],
            continuity_reason_code=(retained.get(album.grouping_key) or (None, None))[
                1
            ],
        )
        for album in proposed
    ]


def _fold_artist_variance(
    groups: dict[tuple[str, str, str], list[GroupingTrack]],
    reasons: dict[tuple[str, str, str], str],
) -> None:
    """Merge same-fold tagged groups whose album artists merely vary.

    M-02 (Phase 1 step 1.7): the partition loop keys on
    ``(directory, album_fold, artist)``, but artist variance alone
    (featured artists, mistags, empty-vs-present) must not shatter one
    album folder. Same-``(directory, album_fold)`` non-empty-artist groups
    therefore merge into the first-sorted one unless the same track number
    is claimed under two distinct non-empty normalized ``album_artist``
    values - two albums sharing one folder, which keeps splitting (golden
    ``same_title_different_album_artists``: both non-empty distinct with a
    track-number collision). Track number 0 is unknown, never a claim, so
    all-unknown groups merge and only known-number collisions split.
    Empty-vs-present splits only on a known-number collision with the
    merge target: non-compilation members of the empty-partition key join
    the merge unless their known track number is already claimed there.
    Compilation members always stay in the empty-partition key (compilation collapse
    unchanged - F-MATCH-06 keeps compilation and plain same-fold tracks as
    two albums). ``feat.``-style containment is not special-cased: with
    distinct track numbers it merges here as one album with
    featured-artist variance (W3 follow-up), splitting only on a
    track-number collision.
    """
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in groups:
        if key[1] and key[0] != "manual":
            buckets[(key[0], key[1])].append(key)
    for (directory, album), bucket_keys in buckets.items():
        # Non-empty partitions never hold compilation members (compilation
        # always partitions ""), so only these keys merge or collide.
        merge_keys = sorted(key for key in bucket_keys if key[2])
        if not merge_keys:
            continue
        claimed: dict[tuple[str, str, str], set[int]] = {
            key: {member.track_number for member in groups[key] if member.track_number}
            for key in merge_keys
        }
        target = merge_keys[0]
        target_claims = set(claimed[target])
        for other in merge_keys[1:]:
            # Only the colliding pair stays split: a group whose known
            # track numbers intersect the accumulated target claims keeps
            # its own key, while disjoint groups still merge in.
            if claimed[other] & target_claims:
                continue
            groups[target].extend(groups.pop(other))
            del reasons[other]
            target_claims |= claimed[other]
        empty_key = (directory, album, "")
        if empty_key in groups:
            absorbable: list[GroupingTrack] = []
            remaining: list[GroupingTrack] = []
            for member in groups[empty_key]:
                if member.is_compilation:
                    remaining.append(member)
                elif member.track_number and member.track_number in target_claims:
                    # Empty-artist absorbs are collision-checked too: a
                    # known number already claimed under the target stays
                    # split instead of duplicating the track number.
                    remaining.append(member)
                else:
                    absorbable.append(member)
                    if member.track_number:
                        target_claims.add(member.track_number)
            if absorbable:
                groups[target].extend(absorbable)
            if remaining:
                groups[empty_key] = remaining
            else:
                del groups[empty_key]
                del reasons[empty_key]


class LocalAlbumGrouper:
    """Create conservative local groups without consulting a provider."""

    def group(
        self,
        tracks: list[GroupingTrack],
        *,
        existing: list[ExistingAlbumMembership] | None = None,
        _cohort_directories_candidate: dict[str, str] | None = None,
    ) -> list[ProposedLocalAlbum]:
        # Pure proposal context only, NEVER authority to persist or lock tracks.
        # Native candidate caller validates the entire binding and fences SQL.
        if _cohort_directories_candidate is not None and (
            set(_cohort_directories_candidate) != {track.local_track_id for track in tracks}
            or len({track.local_track_id for track in tracks}) != len(tracks)
            or len({track.root_id for track in tracks}) != 1
            or len(set(_cohort_directories_candidate.values())) != 1
        ):
            raise ValueError('Complete single-root cohort proposal context required')
        groups: dict[tuple[str, str, str], list[GroupingTrack]] = defaultdict(list)
        reasons: dict[tuple[str, str, str], str] = {}

        for track in sorted(tracks, key=lambda item: item.relative_path):
            if track.membership_locked and track.current_album_id:
                key = ("manual", track.current_album_id, "")
                groups[key].append(track)
                reasons[key] = "MANUAL_MEMBERSHIP_RESTORED"
                continue
            directory, disc = _directory_context(track)
            if _cohort_directories_candidate is not None:
                directory, disc = _cohort_directories_candidate[track.local_track_id], None
            album = normalize_group_value(track.album_title)
            album_artist = normalize_group_value(track.album_artist_name)
            if album:
                artist_partition = "" if track.is_compilation else album_artist
                key = (directory, album, artist_partition)
                groups[key].append(track)
                reasons[key] = (
                    "COMPATIBLE_DISC_DIRECTORIES"
                    if disc is not None
                    else "CONSISTENT_ALBUM_TAGS"
                )
            else:
                key = (directory, "", "")
                groups[key].append(track)
                reasons[key] = "MISSING_ALBUM_TAGS"

        _fold_artist_variance(groups, reasons)
        _fold_top_level_disc_groups(groups, reasons)

        expanded: list[tuple[tuple[str, str, str], list[GroupingTrack], str]] = []
        for key, members in sorted(groups.items()):
            if key[1] or key[0] == "manual":
                reason = reasons[key]
                if key[0] != "manual" and key[1] and _is_parsed_anchored(members):
                    # M-05: the merge is anchored only on parsed evidence
                    # (provisional, low-confidence). Read at append time so
                    # untagged members absorbed below keep a parsed-anchored
                    # target provisional; a genuinely-tagged target keeps
                    # its tagged reason via _is_parsed_anchored.
                    reason = PROVISIONAL_PARSED_GROUP
                expanded.append((key, members, reason))
                continue
            # M-05: the merge target is a tagged-or-parsed group - parsed
            # tracks key as tagged per step 1.5 (parsed substitution in
            # grouping_track_from_row), so a parsed-keyed group qualifies
            # through its non-empty fold exactly like a tagged one.
            # Divergence from legacy_catalog_importer._group_local_only_rows:
            # legacy snapshot rows carry a single album_title column with no
            # raw/tag split and no provenance, so the identical widening is
            # impossible there - legacy keeps its tagged-only rule (see the
            # comment at that site); this grouper is the authority.
            directory_tagged = [
                (candidate_key, candidate_members)
                for candidate_key, candidate_members in groups.items()
                if candidate_key[0] == key[0] and candidate_key[1]
            ]
            if len(directory_tagged) == 1:
                target_key, target_members = directory_tagged[0]
                numbered = all(member.track_number > 0 for member in members)
                occupied = {
                    member.track_number
                    for member in target_members
                    if member.track_number
                }
                agree = all(
                    member.album_title_provenance != "parsed"
                    or normalize_group_value(member.album_title) == target_key[1]
                    for member in members
                )
                # absent/placeholder members keep the strict
                # numbered/no-collision gate; parsed members additionally
                # require fold agreement with the target (a parsed member
                # with an empty value never agrees with a non-empty target
                # fold and stays split - the safe direction for genuinely
                # conflicting parses, which keep per-track split).
                if numbered and agree and not any(
                    member.track_number in occupied for member in members
                ):
                    target_members.extend(members)
                    continue
            for member in members:
                single_key = (key[0], f"untagged:{member.local_track_id}", "")
                expanded.append((single_key, [member], "AMBIGUOUS_FALLBACK_GROUP"))

        proposed: list[ProposedLocalAlbum] = []
        seen_keys: Counter[str] = Counter()
        for key, members, reason in expanded:
            directory = key[1] if key[0] == "manual" else key[0]
            title = _display_consensus(
                [member.album_title for member in members],
                PurePosixPath(directory).name or "Unknown Album",
            )
            artist = _display_consensus(
                [member.album_artist_name for member in members], "Unknown Artist"
            )
            base_key = f"{members[0].root_id}:{directory}:{normalize_group_value(title)}:{normalize_group_value(artist)}"
            seen_keys[base_key] += 1
            grouping_key = (
                base_key
                if seen_keys[base_key] == 1
                else f"{base_key}:{seen_keys[base_key]}"
            )
            proposed.append(
                ProposedLocalAlbum(
                    grouping_key=grouping_key,
                    title=title,
                    album_artist_name=artist,
                    track_ids=sorted(member.local_track_id for member in members),
                    reason_code=reason,
                )
            )
        return assign_album_continuity(existing or [], proposed)
