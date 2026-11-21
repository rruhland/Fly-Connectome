"""Compare control allocation on the same learned gain and coupling estimates."""
import argparse
from pathlib import Path

from .tt17_allocation import AllocatedServo, NativeTravelServo
from .tt17_motor_trial import play


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('case')
    p.add_argument('method',choices=('allocate','travel'))
    a=p.parse_args()
    cls = AllocatedServo if a.method=='allocate' else NativeTravelServo
    play(a.folder,a.case,'bounded_frozen',label=a.method,servo_factory=cls)
