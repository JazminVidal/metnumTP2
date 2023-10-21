import numpy as np
from scipy.stats import pearsonr  
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import subprocess

sns.set_theme()


def write_matrix_to_file(A: np.array, filename: str = "input_data.txt") -> None:
    with open(filename,"w") as f:
        f.write(f"{A.shape[0]} {A.shape[1]}\n")
        np.savetxt(f,A, newline="\n")


def read_matrix_from_file(filename: str = "output_data.txt") -> None:
    return np.loadtxt(filename)


def compile_cpp(filename: str) -> None:
    subprocess.run([
        "g++", 
        filename,
        "-o", 
        "out"
    ])


def eig(A: np.array, num: int = 1, niter: int = 10_000, eps: float = 1e-6) -> tuple[np.array, np.array]:
    write_matrix_to_file(A, "input_data.txt")
    subprocess.run([
        "./out",
        "input_data.txt", 
        "output_eigenvectors.txt",
        "output_eigenvalues.txt",
        str(num),
        str(niter),
        "{:f}".format(eps)
    ])
    V = read_matrix_from_file("output_eigenvectors.txt")
    l = read_matrix_from_file("output_eigenvalues.txt")
    return sorted_eigen(l, V)


def sorted_eigen(l: np.array, V: np.array, rev: bool = True) -> tuple[np.array, np.array]:
    paired_lv = sorted([(l[i], V[:,i]) for i in range(len(l))], key = lambda x: x[0], reverse = rev)
    sorted_l = np.array([pair[0] for pair in paired_lv])
    sorted_V = np.column_stack([pair[1] for pair in paired_lv])
    return sorted_l, sorted_V


def laplacian_matrix(A: np.array) -> np.array:
    D = np.diag(np.sum(A, axis = 1))
    return D - A


def plot_eigenvalues(l: np.array) -> None:
    plt.plot(l, 'o')
    plt.show()


def main() -> None:
    # # Preprocesamiento de los datos
    # X = read_matrix_from_file("redes/ego-facebook.feat")
    # edges = read_matrix_from_file("redes/ego-facebook.edges")
    # np.savetxt("redes/ego-facebook-pre.feat", X[:, 1:], fmt = '%i', newline = "\n")
    # dict_replace = dict(zip(X[:, 0], range(X.shape[0])))
    # norm_edges = np.vectorize(lambda e: dict_replace.get(e, e))(edges)
    # np.savetxt("redes/ego-facebook-pre.edges", norm_edges, fmt = '%i', newline = "\n")
    # A_facebook = np.zeros((X.shape[0], X.shape[0]))
    # for pair in norm_edges:
    #     A_facebook[pair[0], pair[1]] = 1
    #     A_facebook[pair[1], pair[0]] = 1
    # np.savetxt("redes/facebook_matriz.txt", A_facebook, fmt = '%i', newline = "\n")

    # # 1.1.
    # compile_cpp("eig.cpp")


    # 1.2. # TODO: agregar casos de prueba
    A = np.array([
        [ 7,  2,  -3],
        [ 2,  2,  -2],
        [-3, -2,  -2]
    ])
    l, V = eig(A, 3, 10_000, 1e-20)
    for i in range(len(A)):
        assert(np.allclose(A@V, l*V))
        
    D = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])
    v = np.ones((D.shape[0], 1))
    v = v / np.linalg.norm(v)
    B = np.eye(D.shape[0]) - 2 * (v @ v.T) 
    M = B.T @ D @ B
    l, V = eig(M, 5, niter = 10_000, eps = 1e-50)
    assert(np.allclose(M@V, l*V))


    # 2.1.
    A_karate = read_matrix_from_file("redes/karateclub_matriz.txt")
    n_karate = A_karate.shape[0]
    with open("redes/karateclub_labels.txt", "r") as file:
        labels_karate = file.read().split("\n")
        labels_karate = np.array([int(label) for label in labels_karate if label != ""])
    
    # l_A_karate, V_A_karate = eig(A_karate, n_karate, niter = 10_000, eps = 1e-50)
    # np.save("nparrays/eval_karate_A.npy", l_A_karate)
    # np.save("nparrays/evec_karate_A.npy", V_A_karate)
    l_A_karate = np.load("nparrays/eval_karate_A.npy")
    V_A_karate = np.load("nparrays/evec_karate_A.npy")
    l_A_karate_np, V_A_karate_np = np.linalg.eig(A_karate)
    assert(np.allclose(l_A_karate, sorted(l_A_karate_np.real, reverse = True)))
    assert(np.allclose(A_karate@V_A_karate_np, l_A_karate_np*V_A_karate_np))
    v_central_A_karate = V_A_karate[:, 0]
    v_central_A_karate_norm = v_central_A_karate/np.linalg.norm(v_central_A_karate)
    G = nx.from_numpy_array(A_karate)
    pos = nx.spring_layout(G, k = 0.35)
    labels_map = {list(G)[i]: f"{labels_karate[i]}\n{round(v_central_A_karate_norm[i], 2)}" for i in range(len(list(G)))}
    plt.figure(figsize = (12, 6))
    nx.draw_networkx(G, pos, node_color = v_central_A_karate_norm, with_labels = False, node_size = 1000)
    nx.draw_networkx_labels(G, pos, labels = labels_map, font_color = "grey", font_size = 10)
    plt.title("Club de Karate: Centralidad de autovector")
    plt.savefig("img/centralidad_karate.svg", bbox_inches = "tight")
    plt.show()

    # 2.2.
    L_karate = laplacian_matrix(A_karate)
    # l_L_karate, V_L_karate = eig(L_karate, n_karate, niter = 10_000, eps = 1e-50)
    # np.save("nparrays/eval_karate_L.npy", l_L_karate)
    # np.save("nparrays/evec_karate_L.npy", V_L_karate)
    l_L_karate = np.load("nparrays/eval_karate_L.npy")
    V_L_karate = np.load("nparrays/evec_karate_L.npy")
    l_L_karate_np, V_L_karate_np = np.linalg.eig(L_karate)
    assert(np.allclose(l_L_karate, sorted(l_L_karate_np.real, reverse = True)))
    assert(np.allclose(L_karate@V_L_karate_np, l_L_karate_np*V_L_karate_np))

    # TODO: comparar autovectores

    abs_correlations = np.array([abs(pearsonr(V_L_karate[:, i], labels_karate)[0]) for i in range(n_karate)])
    max_correlation_index = abs_correlations.argmax()
    v_division_L_karate = V_L_karate[:, max_correlation_index]
    plt.figure(figsize = (14, 4))
    plt.plot(abs_correlations, "o--")
    plt.plot(max_correlation_index, abs_correlations[max_correlation_index], "ro")
    plt.xticks(ticks = range(0, n_karate), labels = [round(l, 2) for l in l_L_karate], fontsize = 6)
    plt.title("Correlación entre autovectores de la matriz laplaciana y el vector grupo")
    plt.xlabel("Autovalor asociado (adim.)")
    plt.ylabel("Valor absoluto de correlación (adim.)")
    plt.savefig("img/correlacion_karate.svg", bbox_inches = "tight")
    plt.show()

    labels_map = {list(G)[i]: f"{labels_karate[i]}\n{round(v_division_L_karate[i], 2)}" for i in range(len(list(G)))}
    plt.figure(figsize = (12, 6))
    nx.draw_networkx(G, pos, node_color = v_division_L_karate, with_labels = False, node_size = 1000)
    nx.draw_networkx_labels(G, pos, labels = labels_map, font_color = "grey", font_size = 10)
    plt.title("Club de Karate: Vector de máxima correlación con la división en grupos")
    plt.savefig("img/division_karate.svg", bbox_inches = "tight")
    plt.show()


    # 3.1.
    X = read_matrix_from_file("redes/ego-facebook-pre.feat")
    C = X@X.T
    u = 8 # umbral arbitrario de atributos en común
    A_C_facebook = (C >= u).astype(int)
    np.fill_diagonal(A_C_facebook, 0)
    G = nx.from_numpy_array(A_C_facebook)
    pos = nx.spring_layout(G, k = 0.1)
    plt.figure(figsize = (12, 6))
    nx.draw_networkx(G, pos, with_labels = False, node_size = 10)
    plt.title(f"Facebook: Red de amistades determinada por umbral de atributos comunes arbitrario (>= {u})")
    plt.savefig("img/umbral_arbitrario_facebook.svg", bbox_inches = "tight")
    plt.show()
    
    # 3.2.1.
    A_facebook = read_matrix_from_file("redes/facebook_matriz.txt")
    # abs_correlation = abs(pearsonr(A_facebook.flatten(), A_C_facebook.flatten())[0]) # (u = 8; r = 0.06919137913127425)
    
    # 3.2.2.
    facebook_niter = 100
    n_facebook = A_facebook.shape[0]
    # l_A_facebook, V_A_facebook = eig(A_facebook, n_facebook, niter = facebook_niter, eps = 1e-10)
    # l_A_C_facebook, V_A_C_facebook = eig(A_C_facebook, n_facebook, niter = facebook_niter, eps = 1e-10)
    # np.save(f"nparrays/eval_facebook_A_{facebook_niter}.npy", l_A_facebook) # _niter
    # np.save(f"nparrays/evec_facebook_A_{facebook_niter}.npy", V_A_facebook)
    # np.save(f"nparrays/eval_facebook_A_C_{u}_{facebook_niter}.npy", l_A_C_facebook) # _u_niter
    # np.save(f"nparrays/evec_facebook_A_C_{u}_{facebook_niter}.npy", V_A_C_facebook)
    l_A_facebook = np.load(f"nparrays/eval_facebook_A_{facebook_niter}.npy")
    V_A_facebook = np.load(f"nparrays/evec_facebook_A_{facebook_niter}.npy")
    l_A_C_facebook = np.load(f"nparrays/eval_facebook_A_C_{u}_{facebook_niter}.npy")
    V_A_C_facebook = np.load(f"nparrays/evec_facebook_A_C_{u}_{facebook_niter}.npy")
    l_A_facebook_np, V_A_facebook_np = np.linalg.eig(A_facebook)
    # print(list(zip(l_A_facebook, sorted(l_A_facebook_np.real, reverse = True))))
    # assert(np.allclose(l_A_facebook, sorted(l_A_facebook_np.real, reverse = True)))

if __name__ == "__main__":
    main()