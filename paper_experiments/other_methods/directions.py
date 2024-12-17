from torch import Tensor, randn, eye, dtype

from svrz.directions import DirectionGenerator

class GaussianDirections(DirectionGenerator):
    """
    Class representing a direction generator for generating Gaussian directions.

    Summary:
        Generates Gaussian directions by creating a l x d random normal matrix where every entry is sampled from N(0, 1).

    """

    
    def __call__(self) -> Tensor:
        return randn(size=(self.l, self.d), dtype=self.dtype, device=self.device, generator=self.generator)

class SphericalDirections(DirectionGenerator):
    """
    Class representing a direction generator for generating spherical directions.

    Summary:
        Generates spherical directions by creating a random tensor and normalizing columns to unit vectors.
        The generated directions are transposed to switch rows and columns.

    """

    def __call__(self) -> Tensor:
        """
        Method to generate spherical directions by normalizing columns to unit vectors.

        Summary:
            Generates spherical directions by creating a random tensor and normalizing columns to unit vectors.
            The generated directions are transposed to switch rows and columns.

        Returns:
            Tensor: The transposed tensor representing the spherical directions.
        """

        # Generate a random tensor and normalize columns to unit vectors
        P = randn(size=(self.d, self.l), dtype=self.dtype, device=self.device, generator=self.generator)
        P.div_(P.norm(dim=0, p=2))
        return P.T  # Transpose to switch rows and columns
    
    
class CoordinateDirections(DirectionGenerator):
    def __init__(self, d : int, l : int, seed : int, device : str, dtype : dtype) -> None:
        super().__init__(d = d, l = d, seed = seed, device = device, dtype = dtype)    
        self.I = eye(d, device=device, dtype=dtype)
        
    def __call__(self) -> Tensor:
        return self.I
        