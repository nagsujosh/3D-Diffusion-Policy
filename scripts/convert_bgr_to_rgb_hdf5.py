#!/usr/bin/env python3
"""
Convert BGR images to RGB in HDF5 robot demonstration data

This script reads an HDF5 file with BGR images and converts them to RGB format.
It processes both agentview_rgb and eye_in_hand_rgb camera data.
"""

import h5py
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import os


def convert_bgr_to_rgb_hdf5(input_path: str, output_path: str, backup: bool = True):
    """
    Convert BGR images to RGB in HDF5 demonstration data.
    
    Args:
        input_path: Path to input HDF5 file with BGR images
        output_path: Path to output HDF5 file with RGB images
        backup: Whether to create a backup of the original file
    """
    
    # Create backup if requested
    if backup and input_path != output_path:
        backup_path = input_path + '.backup'
        if not os.path.exists(backup_path):
            print(f"Creating backup: {backup_path}")
            shutil.copy2(input_path, backup_path)
            print(f"Backup created successfully")
    
    print(f"Opening input file: {input_path}")
    print(f"Output file: {output_path}")
    
    # Open input file in read mode
    with h5py.File(input_path, 'r') as f_in:
        # Get list of demonstrations
        demo_names = sorted(list(f_in['data'].keys()))
        print(f"Found {len(demo_names)} demonstrations")
        
        # Open output file in write mode
        with h5py.File(output_path, 'w') as f_out:
            # Copy metadata if exists
            if 'mask' in f_in.attrs:
                f_out.attrs['mask'] = f_in.attrs['mask']
            
            # Create data group
            data_out = f_out.create_group('data')
            
            # Process each demonstration
            for demo_name in tqdm(demo_names, desc="Converting demos"):
                demo_in = f_in[f'data/{demo_name}']
                demo_out = data_out.create_group(demo_name)
                
                # Copy attributes
                for attr_name, attr_value in demo_in.attrs.items():
                    demo_out.attrs[attr_name] = attr_value
                
                # Copy actions
                if 'actions' in demo_in:
                    demo_out.create_dataset('actions', data=demo_in['actions'][:])
                
                # Copy rewards if exists
                if 'rewards' in demo_in:
                    demo_out.create_dataset('rewards', data=demo_in['rewards'][:])
                
                # Copy dones if exists
                if 'dones' in demo_in:
                    demo_out.create_dataset('dones', data=demo_in['dones'][:])
                
                # Process observations
                obs_in = demo_in['obs']
                obs_out = demo_out.create_group('obs')
                
                # Copy observation attributes
                for attr_name, attr_value in obs_in.attrs.items():
                    obs_out.attrs[attr_name] = attr_value
                
                # Process each observation type
                for obs_key in obs_in.keys():
                    obs_data = obs_in[obs_key][:]
                    
                    # Convert BGR to RGB for RGB images
                    if 'rgb' in obs_key.lower() and 'depth' not in obs_key.lower():
                        # Check if it's an image with 3 channels
                        if len(obs_data.shape) == 4 and obs_data.shape[-1] == 3:
                            print(f"  Converting {demo_name}/{obs_key}: BGR -> RGB")
                            # BGR to RGB: swap channels 0 and 2
                            obs_data = obs_data[..., ::-1]  # This reverses the last axis (BGR -> RGB)
                    
                    # Save to output
                    obs_out.create_dataset(obs_key, data=obs_data, compression='gzip')
    
    print(f"\nConversion complete!")
    print(f"Output saved to: {output_path}")


def verify_conversion(hdf5_path: str):
    """
    Verify the HDF5 file and show sample RGB values.
    
    Args:
        hdf5_path: Path to HDF5 file to verify
    """
    print(f"\nVerifying: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        demo_names = sorted(list(f['data'].keys()))
        print(f"Demonstrations: {len(demo_names)}")
        
        # Check first demo
        if demo_names:
            demo_name = demo_names[0]
            demo = f[f'data/{demo_name}']
            obs = demo['obs']
            
            print(f"\nSample from {demo_name}:")
            
            # Check agentview_rgb
            if 'agentview_rgb' in obs:
                agentview = obs['agentview_rgb']
                print(f"  agentview_rgb shape: {agentview.shape}")
                sample_pixel = agentview[0, 120, 160]  # Middle pixel of first frame
                print(f"  Sample pixel (frame 0, center): {sample_pixel}")
            
            # Check eye_in_hand_rgb
            if 'eye_in_hand_rgb' in obs:
                eye_in_hand = obs['eye_in_hand_rgb']
                print(f"  eye_in_hand_rgb shape: {eye_in_hand.shape}")
                sample_pixel = eye_in_hand[0, 120, 160]
                print(f"  Sample pixel (frame 0, center): {sample_pixel}")
            
            # List all observation keys
            print(f"\n  Observation keys: {list(obs.keys())}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert BGR images to RGB in HDF5 robot demonstration data')
    
    parser.add_argument('--input', type=str, 
                       default='data/screwdriver_bgr.h5',
                       help='Input HDF5 file path')
    
    parser.add_argument('--output', type=str,
                       default='data/screwdriver.h5',
                       help='Output HDF5 file path')
    
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup of original file')
    
    parser.add_argument('--verify', action='store_true',
                       help='Verify the output file after conversion')
    
    parser.add_argument('--verify-only', type=str, default=None,
                       help='Only verify a specific HDF5 file (skip conversion)')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Only verify mode
        verify_conversion(args.verify_only)
    else:
        # Convert
        try:
            convert_bgr_to_rgb_hdf5(args.input, args.output, backup=not args.no_backup)
            
            # Verify if requested
            if args.verify:
                print("\n" + "=" * 80)
                verify_conversion(args.output)
        
        except FileNotFoundError:
            print(f"ERROR: Input file not found: {args.input}")
        except Exception as e:
            print(f"ERROR: Conversion failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
