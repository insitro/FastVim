# Copyright (C) 2023, Tri Dao.


import warnings

import pytest
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


@pytest.mark.parametrize("wtype", [torch.float32])
@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("seqlen", [4, 6, 128, 254, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize("has_delta_bias", [True])
@pytest.mark.parametrize("delta_softplus", [False])
@pytest.mark.parametrize("has_z", [False])
@pytest.mark.parametrize("has_D", [False, True])
@pytest.mark.parametrize("varBC_groups", [1])  #  B is (B,G=1,N,L) = (B,N,L)
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("compression_factor", [1, 2, 8])
def test_selective_scan(
    is_variable_B,
    is_variable_C,
    varBC_groups,
    has_D,
    has_z,
    has_delta_bias,
    delta_softplus,
    return_last_state,
    seqlen,
    itype,
    wtype,
    compression_factor,
):
    if compression_factor > seqlen:
        return

    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = "cuda"
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 2
    # dim = 4
    batch_size = dim = 1
    dstate = 8
    is_complex = wtype == torch.complex64

    if seqlen % compression_factor != 0:
        warnings.warn("Seqlen must be evenly divisible by compression factor")
        return
    seqlen_c = seqlen // compression_factor

    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen_c if not is_complex else seqlen_c * 2)
    else:
        B_shape = (
            batch_size,
            varBC_groups,
            dstate,
            seqlen_c if not is_complex else seqlen_c * 2,
        )
    B = torch.randn(
        *B_shape,
        device=device,
        dtype=wtype if not is_variable_B else itype,
        requires_grad=True,
    )
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen_c if not is_complex else seqlen_c * 2)
    else:
        C_shape = (
            batch_size,
            varBC_groups,
            dstate,
            seqlen_c if not is_complex else seqlen_c * 2,
        )
    C = torch.randn(
        *C_shape,
        device=device,
        dtype=wtype if not is_variable_C else itype,
        requires_grad=True,
    )
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_z:
        raise ValueError("No Z")
        z = torch.randn(
            batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True
        )
    else:
        z = None
    if has_delta_bias:
        delta_bias = (
            0.5 * torch.rand(dim, device=device, dtype=torch.float32)
        ).requires_grad_()
    else:
        delta_bias = None

    # Generate u and compressed u
    u = torch.randn(
        batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True
    )

    u_compressed = u.reshape((batch_size, dim, compression_factor, seqlen_c)).mean(
        dim=2
    )

    delta = (
        0.5 * torch.rand(batch_size, dim, seqlen_c, device=device, dtype=itype)
    ).requires_grad_()

    print("Contiguous?? ", delta.is_contiguous())
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    u_c_ref = u_compressed.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = (
        delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    )

    out, *rest = selective_scan_fn(
        u,
        u_compressed,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )
    if return_last_state:
        state = rest[0]

    out_ref, *rest = selective_scan_ref(
        u_ref,
        u_c_ref,
        delta_ref,
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        z=z_ref,
        delta_bias=delta_bias_ref,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )
    if return_last_state:
        state_ref = rest[0]
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u
    out_ref_decomp = out_ref

    print(out, out.shape)
    print(out_ref_decomp, out_ref.shape)

    print(f"Output max diff: {(out - out_ref_decomp).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref_decomp).abs().mean().item()}")
    print(f"Shape out {out.shape} ref {out_ref_decomp.shape}")

    # print(out.tolist())
    assert torch.allclose(out, out_ref_decomp, rtol=rtol, atol=atol)
    if return_last_state:
        print(f"State max diff: {(state - state_ref).abs().max().item()}")
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)

    # Skip backwards for now
    return


#    g = torch.randn_like(out)
#    out_ref.backward(g)
#    out.backward(g)
#
#    print(f"du max diff: {(u.grad - u_ref.grad).abs().max().item()}")
#    print(f"ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}")
#    print(f"dA max diff: {(A.grad - A_ref.grad).abs().max().item()}")
#    print(f"dB max diff: {(B.grad - B_ref.grad).abs().max().item()}")
#    print(f"dC max diff: {(C.grad - C_ref.grad).abs().max().item()}")
#    if has_D:
#        print(f"dD max diff: {(D.grad - D_ref.grad).abs().max().item()}")
#    if has_z:
#        print(f"dz max diff: {(z.grad - z_ref.grad).abs().max().item()}")
#    if has_delta_bias:
#        print(
#            f"ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}"
#        )
#
#    assert torch.allclose(
#        u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2
#    )
#    assert torch.allclose(
#        delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10
#    )
#    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
#    assert torch.allclose(
#        B.grad,
#        B_ref.grad,
#        rtol=rtolw if not is_variable_B else rtol,
#        atol=atolw if not is_variable_B else atol,
#    )
#    assert torch.allclose(
#        C.grad,
#        C_ref.grad,
#        rtol=rtolw if not is_variable_C else rtol,
#        atol=atolw if not is_variable_C else atol,
#    )
#    if has_D:
#        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
#    if has_z:
#        assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
#    if has_delta_bias:
#        assert torch.allclose(
#            delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw
#        )
