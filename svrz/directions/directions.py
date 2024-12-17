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
        self.generator = Generator(device=device)
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


