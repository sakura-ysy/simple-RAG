
def combine_input_prompt_chunks(
        prompt_chunks: list[str],
    ) -> str:
    """Combine the input chunks by adding the special separators in between

    :param prompt_chunks: A list of input chunks in string
    :type prompt_chunks: List[str]

    :return: The combined input tensor
    :rtype: torch.Tensor
    """
    separator = " # #<s> "
    return separator.join(prompt_chunks)