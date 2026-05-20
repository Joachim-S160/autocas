"""Combine active spaces and occupation lists.

These functions allow the combination of multiple active spaces along a reaction coordinate through orbital maps.
"""
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details. """

from typing import Dict, List, Optional, Tuple

import numpy as np


def transform_orbital_groups(orbital_groups: List[List[List[int]]]):
    """Transform the orbital mapping in terms of groups to a map of orbitals to groups.

    Parameters
    ----------
        orbital_groups : List[List[List[int]]]
            The list of orbital groups.

    Returns
    -------
        A matrix (orbitals x systems) containing the group indices for each orbital.
    """
    n_systems = len(orbital_groups[0])
    n_orbitals = 0
    for group in orbital_groups:
        n_orbitals += len(group[0])
    orbital_to_group = np.zeros((n_orbitals, n_systems), dtype=int)  # type: ignore
    for i_group, group in enumerate(orbital_groups):
        for i_sys, sys in enumerate(group):
            for i_orb in sys:
                orbital_to_group[i_orb, i_sys] = i_group
    return orbital_to_group


def combine_active_spaces(occupations: List[List[int]], active_spaces: List[List[int]],
                          orbital_groups: List[List[List[int]]],
                          initial_valence_occupations: Optional[List[List[int]]] = None,
                          initial_valence_indices: Optional[List[List[int]]] = None
                          ) -> Tuple[List[List[int]], List[List[int]]]:
    """Combine multiple active spaces (e.g., from points along a reaction coordinate) with an orbital map.

    Parameters
    ----------
    occupations : List[List[int]]
        The occupations for each active space and each orbital.
        E.g., [[2, 2, 0, 0], [2, 2, 0, 0]] for two systems.
    active_spaces : List[List[int]]
        The indices of the orbitals considered as active (starting from 0) for each system.
    orbital_groups : List[List[List[int]]]
        Group-wise mapping of the orbitals. The orbitals are grouped into sets that are indistinguishable.
        For each group, and system the orbital indices are given, e.g.,
        [
        [[3, 4, 5], [3, 4, 6]],
        [[6], [5]],
        [[7], [7]],
        ...
        ]
        This list means that the orbitals 3, 4, and 5 of the first system are mapped to the orbitals 3, 4, and 6 of
        the second system. The orbital 6 of system 1 is mapped to orbital 5 of system 2, and the orbital 7 of system
        1 is mapped to the orbital 7 of system 2.
    initial_valence_occupations : Optional[List[List[int]]]
        Per-system ROHF occupations of the initial valence CAS (open-shell only).
        Used as fallback when an orbital is added to the union via alignment but
        was not in that system's per-geom CAS. Without this, fallback was
        ``max(known)`` which mis-attributes a SOMO occupation to a DOMO orbital
        (or vice versa) when an alignment crosses occupation type, producing an
        active space whose total electron count is incompatible with the spin
        multiplicity (odd NACTEL with quintet, etc.).
    initial_valence_indices : Optional[List[List[int]]]
        Per-system orbital indices corresponding to ``initial_valence_occupations``.

    Returns
    -------
    The combined active spaces according to mapping.
    """
    # orbital_groups index 0: orbital group
    #                      1: system index
    #                      2: orbital index
    orbital_to_group = transform_orbital_groups(orbital_groups)
    n_systems = len(orbital_groups[0])

    # Build per-system orbital→occupation lookup from the initial valence CAS.
    # If not provided we keep the legacy behaviour (max of known across systems).
    initial_lookups: List[Dict[int, int]] = []
    if initial_valence_occupations is not None and initial_valence_indices is not None:
        for occ_list, idx_list in zip(initial_valence_occupations, initial_valence_indices):
            initial_lookups.append({int(idx): int(occ) for idx, occ in zip(idx_list, occ_list)})

    # Key: (group_index, position_within_group) → per-system occupation list.
    # Each system records its own occupation rather than a global max so that
    # orbital crossings between geometries (e.g. σ↔σ* at stretched vs. compressed)
    # do not inflate the electron count in the combined active space.
    active_slot_occs: Dict[Tuple[int, int], List[Optional[int]]] = {}
    for i_sys, (occupation, active_space) in enumerate(zip(occupations, active_spaces)):
        for i_orb, occ in zip(active_space, occupation):
            i_group = int(orbital_to_group[i_orb, i_sys])
            i_pos = orbital_groups[i_group][i_sys].index(i_orb)
            # When any orbital in a group is first encountered, initialise ALL
            # position slots for that group so that partners of the active
            # orbital are included in the union even if they never appear in
            # any per-geom active_space directly.
            group_size = len(orbital_groups[i_group][0])
            for i_pos_all in range(group_size):
                if (i_group, i_pos_all) not in active_slot_occs:
                    active_slot_occs[(i_group, i_pos_all)] = [None] * n_systems
            active_slot_occs[(i_group, i_pos)][i_sys] = occ
    # Build combined active spaces: each system uses its own occupation;
    # for systems where the orbital was not in the per-geom CAS, fall back to
    # the system's ROHF reference occupation (initial valence CAS) when
    # available, otherwise to the max of known occupations.
    #
    # Group-completion slots (known=[]) are orbitals autoCAS never selected at
    # any geometry. For UHF open-shell calculations (initial_lookups non-empty),
    # skip all such slots: alpha and beta spin-orbitals occupy the same IBO
    # alignment group but are structurally distinct — completing the unselected
    # spin partner is physically wrong. Group-completion is only meaningful for
    # RHF/ROHF spatial orbital degenerate subspaces (no initial_lookups).
    new_active_spaces: List[List[int]] = [[] for _ in range(n_systems)]
    new_occupations: List[List[int]] = [[] for _ in range(n_systems)]
    for (i_group, i_pos), occ_list in active_slot_occs.items():
        known = [o for o in occ_list if o is not None]
        if not known:
            if initial_lookups:
                continue
        max_known = max(known) if known else 0
        for i_sys, sys_orbitals in enumerate(orbital_groups[i_group]):
            orb_idx = sys_orbitals[i_pos]
            new_active_spaces[i_sys].append(orb_idx)
            if occ_list[i_sys] is not None:
                new_occupations[i_sys].append(occ_list[i_sys])
            elif initial_lookups and orb_idx in initial_lookups[i_sys]:
                new_occupations[i_sys].append(initial_lookups[i_sys][orb_idx])
            else:
                new_occupations[i_sys].append(max_known)
    return new_occupations, new_active_spaces
