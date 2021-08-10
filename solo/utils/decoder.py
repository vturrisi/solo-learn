import torch


class BitShiftDecoder:
    @staticmethod
    def int_to_binary(x: torch.Tensor, bits: int) -> torch.Tensor:
        """Converts a Tensor of integers to a Tensor in binary format.
        https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

        Args:
            x (torch.Tensor): tensor of interges to convert to binary format.
            bits (torch.Tensor): number of bits to use.

        Returns:
            torch.Tensor: x in binary format.
        """

        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    @staticmethod
    def binary_to_int(b: torch.Tensor, bits: int) -> torch.Tensor:
        """Converts a Tensor in binary format to a Tensor of integers.
        https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

        Args:
            b (torch.Tensor): tensor of binary data to convert to integer.
            bits (torch.Tensor): number of bits.

        Returns:
            torch.Tensor: x in integer format.
        """

        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1).detach()

    def decode_targets(self, encoded):
        binary_encoded = self.int_to_binary(encoded, 32)
        targets = self.binary_to_int(binary_encoded[:, 1:-21], 10).long()
        return targets

    def decode_indexes(self, encoded):
        binary_encoded = self.int_to_binary(encoded, 32)
        indexes = self.binary_to_int(binary_encoded[:, -21:], 21).long()
        return indexes
