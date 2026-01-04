#!/usr/bin/env python
"""
Command-line interface for computing stability bounds L and U.
"""

import argparse
import json
import sys
from src.compute_bounds import compute_LU, check_stability_for_c


def main():
    parser = argparse.ArgumentParser(
        description="Compute stability bounds L and U for the ACE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --a_tilde 1.0 --b 0.2 --gamma 0.5 --p_star 0.7
  %(prog)s --check 1.0 --a_tilde 1.0 --b 0.2 --gamma 0.5 --p_star 0.7
  %(prog)s --json --a_tilde 1.0 --b 0.2 --gamma 0.5 --p_star 0.7
        """
    )
    
    # Model parameters
    parser.add_argument('--a_tilde', type=float, default=1.0,
                       help='Effective extraction rate (default: 1.0)')
    parser.add_argument('--b', type=float, default=0.2,
                       help='Cooperation cost coefficient (default: 0.2)')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Responsibility benefit coefficient (default: 0.5)')
    parser.add_argument('--p_star', type=float, default=0.7,
                       help='Architect symmetry parameter (default: 0.7)')
    
    # Optional: check specific c value
    parser.add_argument('--check', type=float, metavar='C',
                       help='Check stability for specific c value')
    
    # Output options
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Compute bounds
        L, U, info = compute_LU(
            a_tilde=args.a_tilde,
            b=args.b,
            gamma=args.gamma,
            p_star=args.p_star
        )
        
        if args.check is not None:
            # Check stability for specific c
            result = check_stability_for_c(
                c=args.check,
                a_tilde=args.a_tilde,
                b=args.b,
                gamma=args.gamma,
                p_star=args.p_star
            )
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\nStability check for c = {args.check}:")
                print(f"  L = {L:.6f}")
                print(f"  U = {U:.6f}")
                print(f"  Stable: {result['stable']}")
                print(f"  Margin: {result['stability_margin']:.6f}")
                if args.verbose:
                    print(f"  Distance to L: {result['distance_to_bounds']['to_L']:.6f}")
                    print(f"  Distance to U: {result['distance_to_bounds']['to_U']:.6f}")
        
        else:
            # Just compute bounds
            if args.json:
                output = {
                    'L': L,
                    'U': U,
                    **info
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"\nStability bounds for:")
                print(f"  ã = {args.a_tilde}, b = {args.b}, γ = {args.gamma}, p* = {args.p_star}")
                print(f"\n  L = {L:.6f}")
                print(f"  U = {U:.6f}")
                print(f"  Width = {U - L:.6f}")
                
                if args.verbose:
                    print(f"\nAdditional information:")
                    print(f"  Interval non-empty: {info['nonempty']}")
                    print(f"  Limits as p→0: L→{info['limits']['p→0']['L']}, U→{info['limits']['p→0']['U']}")
                    print(f"  Limits as p→1: L→{info['limits']['p→1']['L']}, U→∞")
    
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()