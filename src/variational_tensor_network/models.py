"""Define two-site Hamiltonians and their model metadata."""

from torch import Tensor, eye, kron

def make_hamil_term(S_alpha, s1, S_beta, s2, pf, L):
    if s2 > s1:
        print(f"Warning: Swapping sites {s1} and {s2} to ensure s1 > s2")
        S_alpha, s1, S_beta, s2 = S_beta, s2, S_alpha, s1

    if s1 > L or s2 > L:
        raise ValueError("Sites must be less than or equal to L")

    I = eye

    A = I(2 ** (s1 - 1))
    B = S_alpha
    C = I(2 ** (s2 - s1 - 1))
    D = S_beta
    E = I(2 ** (L - s2))

    term = pf * kron(kron(kron(kron(A, B), C), D), E)
    return term

def two_site_heisenberg_term(L: int, J: float = 1.0, site1: int = 1, site2: int = 2) -> Tensor:
    # L is the number of sites, J is the coupling constant. 
    # Return the spin-1/2 Heisenberg Hamiltonian on sites1 and site2 for a L-site system.
    if L < 2:
        raise ValueError("L must be at least 2")

    S_z = Tensor([[0.5, 0.0], [0.0, -0.5]])
    S_plus = Tensor([[0.0, 1.0], [0.0, 0.0]])
    S_minus = S_plus.mT.contiguous()
    H = eye(2 ** L) * 0.0

    H += make_hamil_term(S_z, site1, S_z, site2, J, L)
    H += make_hamil_term(S_plus, site1, S_minus, site2, J / 2, L)
    H += make_hamil_term(S_minus, site1, S_plus, site2, J / 2, L)

    return H

def single_site_Zeeman_term(L: int, h: float = 1.0, site: int = 1) -> Tensor:
    # L is the number of sites, h is the magnetic field.
    # Return the spin-1/2 Zeeman Hamiltonian on site for a L-site system.
    if L < 1:
        raise ValueError("L must be at least 1")

    S_z = Tensor([[0.5, 0.0], [0.0, -0.5]])
    H = eye(2 ** L) * 0.0

    H += make_hamil_term(S_z, site, S_z, site, h, L)

    return H


def heisenberg_bond(J: float = 1.0) -> Tensor:
    """Uses the product basis (↑↑,↑↓,↓↑,↓↓)."""
    H = Tensor([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  2.0, 0.0],
        [0.0,  2.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ])
    return (J / 4) * H


def ffb_bond(JD: float = 2.0, J: float = 1.0, h: float = 0) -> Tensor:
    """Use the product basis (↑↑, ↑↓, ↓↑, ↓↓) for each dimer."""
    I = eye(4)
    Tz = Tensor([
        [1.0, 0.0, 0.0,  0.0],
        [0.0, 0.0, 0.0,  0.0],
        [0.0, 0.0, 0.0,  0.0],
        [0.0, 0.0, 0.0, -1.0],
    ])
    Tp = Tensor([
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    Tm = Tp.mT.contiguous()
    H_Heis = heisenberg_bond()
    H_JD = kron(H_Heis, I) + kron(I, H_Heis)
    H_J   = kron(Tz, Tz) + (1/2) * (kron(Tp, Tm) + kron(Tm, Tp))
    H_h   = kron(Tz, I) + kron(I, Tz)
    return (JD/4) * H_JD + J * H_J + (h/4) * H_h

def heisenberg_bond_alternate(J: float = 1.0, h: float = 0) -> Tensor:
    #Constructs the Heisenberg bond term with Zeeman term for a two-site system
    H =  two_site_heisenberg_term(L=2, J=J, site1=1, site2=2)
    H += single_site_Zeeman_term(L=2, h=h/2, site=1) #removing the double counting of the Zeeman term by dividing h by 2 for each site
    H += single_site_Zeeman_term(L=2, h=h/2, site=2) #removing the double counting of the Zeeman term by dividing h by 2 for each site
    return  H

def ffb_bond_alternate(JD: float = 2.0, J: float = 1.0, h: float = 0) -> Tensor:
    #Constructs the Heisenberg Hamiltonian for fully frustrated bilayer model term
    #     4 
    #    ||   
    #     3  
    #     
    #  1 == 2
    #Dimer terms
    H =  two_site_heisenberg_term(L=4, J=JD/2, site1=1, site2=2) #removing the double counting of the dimer term by dividing JD by 2 for each dimer
    H += two_site_heisenberg_term(L=4, J=JD/2, site1=3, site2=4) #removing the double counting of the dimer term by dividing JD by 2 for each dimer
    #inter-dimer terms
    H += two_site_heisenberg_term(L=4, J=J, site1=1, site2=3)
    H += two_site_heisenberg_term(L=4, J=J, site1=2, site2=3)
    H += two_site_heisenberg_term(L=4, J=J, site1=1, site2=4) 
    H += two_site_heisenberg_term(L=4, J=J, site1=2, site2=4)
    # removing the last two terms implements Shastry-Sutherland model, which is a special case of the fully frustrated bilayer model
    #Zeeman terms
    H += single_site_Zeeman_term(L=4, h=h/2, site=1) #removing the double counting of the Zeeman term by dividing h by 2 for each site
    H += single_site_Zeeman_term(L=4, h=h/2, site=2) #removing the double counting of the Zeeman term by dividing h by 2 for each site
    H += single_site_Zeeman_term(L=4, h=h/2, site=3) #removing the double counting of the Zeeman term by dividing h by 2 for each site
    H += single_site_Zeeman_term(L=4, h=h/2, site=4) #removing the double counting of the Zeeman term by dividing h by 2 for each site
    return  H


MODELS = {
    "heisenberg": {
        "bond": heisenberg_bond,
        "d": 2,
        "label": "Heisenberg",
    },
    "heisenberg_alternate": {
        "bond": heisenberg_bond_alternate,
        "d": 2,
        "label": "Heisenberg (Alternate)",
    },
    "ffb": {
        "bond": ffb_bond,
        "d": 4,
        "label": "FFB",
    },
    "ffb_alternate": {
            "bond": ffb_bond_alternate,
            "d": 4,
            "label": "FFB (Alternate)",
        },
}
