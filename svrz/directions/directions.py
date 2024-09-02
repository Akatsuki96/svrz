from torch import Generator, Tensor, dtype, randn, eye, randperm, float32
from torch.linalg import qr


class DirectionGenerator:
    """
    Class representing a generic direction generator.

    Summary:
        Initializes a direction generator with dimensions, seed, device, and data type.
        Provides the shape property and a placeholder call method to be implemented in subclasses.

    Attributes:
        d: The number of rows in the generated directions.
        l: The number of columns in the generated directions.
        dtype: The data type of the generated directions.
        device: The device on which the directions are generated.
        generator: The random number generator used for seeding.

    Properties:
        shape: Returns a tuple representing the shape of the generated directions.

    Methods:
        __call__: Placeholder method that raises NotImplementedError. To be implemented in subclasses.

    Raises:
        NotImplementedError: If the __call__ method is not implemented in subclasses.
    """

    def __init__(self, d : int, l : int, seed : int = 1231415, device : str = 'cpu', dtype : dtype = float32) -> None:
        self.d = d
        self.l = l
        self.dtype = dtype
        self.device = device
        self.generator = Generator()
        self.generator.manual_seed(seed)

    @property
    def shape(self):
        """
        Property representing the shape of the generated directions.

        Summary:
            Returns a tuple representing the shape of the generated directions with the number of columns followed by the number of rows.

        Returns:
            Tuple: A tuple representing the shape of the generated directions.
        """

        return (self.l, self.d)
        
    def __call__(self) -> Tensor:
        raise NotImplementedError("Call method is implemented in subclasses!")


class GaussianDirections(DirectionGenerator):
    
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
    
class QRDirections(DirectionGenerator):
    """
    Class representing a direction generator for generating directions using QR decomposition.

    Summary:
        Generates directions by performing QR decomposition on a random tensor.
        The Q matrix from the decomposition is transposed to switch rows and columns.
    """

    def __call__(self) -> Tensor:
        """
        Method to generate directions using QR decomposition.

        Summary:
            Generates directions by performing QR decomposition on a random tensor.
            The Q matrix from the decomposition is transposed to switch rows and columns.

        Returns:
            Tensor: The transposed Q matrix representing the generated directions.
        """

        P = randn(size=(self.d, self.l), dtype =self.dtype, device=self.device, generator=self.generator)
        return qr(P, mode='reduced')[0].T


class HouseholderDirections(DirectionGenerator):
    
    def __call__(self) -> Tensor:
        v = randn(size=(self.d,), dtype=self.dtype, device=self.device, generator=self.generator)
        v.div_(v.norm(p=2))
        return (eye(self.d, self.l, dtype=self.dtype, device=self.device) - 2 * v.outer(v[:self.l])).T


class CoordinateDirections(DirectionGenerator):
    """
    Class representing a direction generator for generating coordinate directions.

    Summary:
        Generates coordinate directions based on the dimensions and random indices.
        If the number of rows equals the number of columns, it returns the identity matrix; otherwise, it selects columns based on random indices.

    """

    def __init__(self, d : int, l : int, seed : int, device : str, dtype : dtype) -> None:
        super().__init__(d=d, l=l, seed=seed, device=device, dtype=dtype)    
        self.I = eye(d, device=device, dtype=dtype)
        
    def __call__(self) -> Tensor:
        """
        Method to generate coordinate directions based on dimensions and random indices.

        Summary:
            Generates coordinate directions by selecting columns from the identity matrix based on random indices.
            If the number of rows equals the number of columns, it returns the identity matrix; otherwise, it selects columns based on random indices.

        Returns:
            Tensor: The transposed tensor representing the generated coordinate directions.
        """

        if self.d == self.l:
            return self.I
        
        indices = randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        return self.I[:, indices].T
        