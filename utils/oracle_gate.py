from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def video_visibility(video_name):
    """Return the hard Oracle Gate value encoded by a video name.

    Video names end in ``_clear`` or ``_vague``.  Unknown suffixes are
    rejected so a malformed sample cannot silently receive the wrong gate.
    """
    stem = os.path.splitext(os.path.basename(str(video_name).rstrip('/\\')))[0]
    visibility = stem.rsplit('_', 1)[-1].lower()
    if visibility == 'clear':
        return 0.0
    if visibility == 'vague':
        return 1.0
    raise ValueError(
        'video name must end in _clear or _vague, got {!r}'.format(video_name))
