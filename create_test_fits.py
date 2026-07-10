import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def create_synthetic_fits(filename="sample.fits"):
    # Generate 512x512 synthetic image with a Bayer-like grid pattern and a simulated star
    width, height = 512, 512
    data = np.zeros((height, width), dtype=np.uint16)
    
    # Draw a simulated Gaussian star in the center
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    r2 = (x - cx)**2 + (y - cy)**2
    star = np.exp(-r2 / (2.0 * 15.0**2)) * 40000.0  # Max intensity around 40000
    
    # Overlay star
    data = (data + star).astype(np.uint16)
    
    # Add a Bayer pattern grid (RGGB color structure) to make debayering noticeable
    # R  G
    # G  B
    for r in range(height):
        for c in range(width):
            val = data[r, c]
            if r % 2 == 0:
                if c % 2 == 0:
                    data[r, c] = min(65535, int(val * 1.0))
                else:
                    data[r, c] = min(65535, int(val * 0.6))
            else:
                if c % 2 == 0:
                    data[r, c] = min(65535, int(val * 0.6))
                else:
                    data[r, c] = min(65535, int(val * 0.3))
                    
    # Add some background noise
    noise = np.random.normal(500, 100, size=(height, width))
    data = np.clip(data + noise, 0, 65535).astype(np.uint16)

    # Set up WCS coordinates
    w = WCS(naxis=2)
    w.wcs.crpix = [cx, cy]
    w.wcs.cdelt = [-0.000277777777778, 0.000277777777778] # 1 arcsec/pixel
    w.wcs.crval = [83.633083, 22.0145] # Crab Nebula coordinates (RA/DEC)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    header = w.to_header()
    header['OBJECT'] = 'Synthetic Star'
    header['EXPTIME'] = 300.0
    header['INSTRUME'] = 'Simulated Camera'
    header['BAYERPAT'] = 'RGGB'
    
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filename, overwrite=True)
    print(f"Created synthetic FITS: {filename}")

if __name__ == "__main__":
    create_synthetic_fits()
