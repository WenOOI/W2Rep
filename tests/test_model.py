import torch

from w2rep.models.w2rep import W2RepEncoder, W2RepPredictor


def test_encoder_image_and_video_shapes():
    encoder = W2RepEncoder(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        z_tokens=4,
        max_frames=3,
    ).eval()
    pixels = torch.randn(2, 3, 3, 32, 32)
    mask = torch.tensor([[0, 2, 5, 7], [1, 3, 8, 10]])
    output = encoder(pixels, mask, with_z=True)
    assert output["patch_out"].shape == (2, 3, 4, 32)
    assert output["z_out"].shape == (2, 4, 32)
    image_output = encoder(pixels[:, 0], with_z=False)
    assert image_output["patch_out"].shape == (2, 1, 16, 32)
    assert image_output["z_out"] is None


def test_predictor_shape_and_gradients():
    predictor = W2RepPredictor(
        embed_dim=32,
        pred_dim=16,
        depth=2,
        num_heads=4,
        grid=4,
        z_dim=32,
    )
    context = torch.randn(2, 5, 32, requires_grad=True)
    context_positions = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
    target_positions = torch.tensor([[8, 9, 10], [0, 1, 2]])
    offset = torch.tensor([-2.0, 3.0])
    z = torch.randn(2, 4, 32, requires_grad=True)
    output = predictor(context, context_positions, target_positions, offset, z)
    assert output.shape == (2, 3, 32)
    output.square().mean().backward()
    assert context.grad is not None
    assert z.grad is not None


def test_intermediate_features_support_non_square_inputs():
    encoder = W2RepEncoder(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=4,
        num_heads=4,
        z_tokens=2,
    ).eval()
    pixels = torch.randn(1, 3, 32, 48)
    features, grid = encoder.forward_intermediates(pixels, (0, 3))
    assert grid == (4, 6)
    assert [feature.shape for feature in features] == [(1, 24, 32), (1, 24, 32)]
    final = encoder(pixels, with_z=False)["patch_out"][:, 0]
    torch.testing.assert_close(features[-1], final)
