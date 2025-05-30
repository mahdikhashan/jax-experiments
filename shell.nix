{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.uv
  ];

  shellHook = ''
    uv --version
    uv add scikit-learn
    uv add torch torchvision
    uv add matplotlib
    uv add ipykernel
  '';
}