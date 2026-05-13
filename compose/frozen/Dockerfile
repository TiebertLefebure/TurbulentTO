# Legacy FEniCS (dolfin) + python3 als PID 1 — geen Phusion my_init (shutdown-bug onder Py3).
#
# quay.io/fenicsproject/stable:* gebruikt manifest schema 1; Docker 25+ weigert dat.
# fenics/stable:py3 op Docker Hub is de gangbare vervanger (zelfde soort stack).
FROM --platform=linux/amd64 fenics/stable:py3
ENTRYPOINT ["python3"]
