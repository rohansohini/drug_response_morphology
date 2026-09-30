def water_fill_allocate(group_sizes: dict, budget: int) -> dict:
    '''
    Distribute an integer budget as evenly as possible across groups,
    never allocating more to a group than it has available. Groups with
    fewer than an equal share get everything they have; the leftover
    budget is redistributed evenly among the remaining groups, repeating
    until the budget is exhausted or every group is fully allocated.

    Inputs:
        group_sizes: {group_key: n_available}, n_available >= 0
        budget:      total number of items to allocate across all groups
                     (budget >= 0)
    Outputs:
        allocation: {group_key: n_allocated} for every key in group_sizes,
                    0 <= n_allocated <= n_available, and
                    sum(allocation.values()) == min(budget, sum(group_sizes.values()))
    '''
    allocation = {k: 0 for k in group_sizes}
    remaining  = dict(group_sizes)
    budget_left = budget

    while remaining and budget_left > 0:
        share = budget_left // len(remaining)

        if share == 0:
            # not enough budget left for a full unit per group - hand out
            # one at a time, smallest-available groups first, so scarce
            # budget isn't wasted on groups that would've been capped anyway
            for k in sorted(remaining, key=lambda k: remaining[k]):
                if budget_left == 0:
                    break
                allocation[k] += 1
                remaining[k]  -= 1
                budget_left   -= 1
                if remaining[k] == 0:
                    del remaining[k]
            break

        capped = {k: n for k, n in remaining.items() if n <= share}

        if not capped:
            # every remaining group can absorb a full equal share
            for k in remaining:
                allocation[k] += share
                budget_left   -= share
            # leftover from integer division (< len(remaining)) goes one-by-one
            for k in list(remaining)[:budget_left]:
                allocation[k] += 1
                budget_left   -= 1
            break

        for k, n in capped.items():
            allocation[k] += n
            budget_left   -= n
            del remaining[k]

    return allocation
